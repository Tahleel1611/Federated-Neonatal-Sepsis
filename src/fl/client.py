from __future__ import annotations

import argparse
from pathlib import Path

import flwr as fl
import numpy as np
import pandas as pd
import torch
from opacus.validators import ModuleValidator
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import DataConfig, FederatedConfig, ModelConfig, TrainConfig
from src.constants import MODEL_FEATURE_COLUMNS, TARGET_COLUMN
from src.data.federated.partition import partition_by_hospital
from src.data.loaders.synthetic_loader import generate_synthetic_nicu_logs
from src.data.preprocess.pipeline import preprocess_nicu_data
from src.eval.metrics import binary_metrics, save_precision_recall_curve, select_threshold_for_target_sensitivity
from src.models.transformer_lstm import TransformerLSTMSepsisModel
from src.privacy.dp import make_private_with_opacus
from src.privacy.smpc_interface import SMPCAdapter


def load_data(data_cfg: DataConfig):
    """Return a raw NICU DataFrame from MIMIC-III or synthetic generator."""
    if data_cfg.use_mimic:
        if data_cfg.mimic_prebuilt_csv and Path(data_cfg.mimic_prebuilt_csv).exists():
            return pd.read_csv(data_cfg.mimic_prebuilt_csv)
        from src.data.loaders.mimic_loader import build_mimic_nicu_dataset
        return build_mimic_nicu_dataset(
            mimic_dir=data_cfg.mimic_dir,
            prediction_horizon_h=data_cfg.prediction_horizon_hours,
            max_stays=data_cfg.mimic_max_stays,
        )
    return generate_synthetic_nicu_logs(data_cfg)


def weighted_bce_loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=pred.device, dtype=pred.dtype)
    )
    return criterion(pred, target)


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.80, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        loss = alpha_t * torch.pow((1.0 - pt), self.gamma) * bce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def build_sequences(df, seq_len_steps: int = 12):
    feature_cols = MODEL_FEATURE_COLUMNS
    x_list, y_list = [], []
    for _, group in df.groupby("Patient_ID", sort=False):
        values = group[feature_cols + [TARGET_COLUMN]].to_numpy(dtype=float)
        for idx in range(seq_len_steps, len(values)):
            x_list.append(values[idx - seq_len_steps : idx, :-1])
            y_list.append(values[idx, -1])
    x = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return x, y


def split_by_patient_with_min_positives(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    min_val_positive_patients: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    if "Patient_ID" not in df.columns:
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
        return train_df, val_df, 0, 0

    patient_labels = (
        df.groupby("Patient_ID", sort=False)[TARGET_COLUMN]
        .max()
        .fillna(0.0)
        .astype(float)
    )
    all_patients = patient_labels.index.to_numpy()
    if len(all_patients) < 2:
        return df.copy(), df.iloc[0:0].copy(), int(patient_labels.sum()), 0

    pos_patients = patient_labels[patient_labels > 0].index.to_numpy()
    neg_patients = patient_labels[patient_labels <= 0].index.to_numpy()

    n_val = max(1, int(round(len(all_patients) * float(test_size))))
    rng = np.random.default_rng(seed)

    if len(pos_patients) > 0:
        target_pos_val = max(min_val_positive_patients, int(round(n_val * len(pos_patients) / len(all_patients))))
        if len(pos_patients) > 1:
            target_pos_val = min(target_pos_val, len(pos_patients) - 1)
        else:
            target_pos_val = min(target_pos_val, len(pos_patients))
    else:
        target_pos_val = 0

    target_neg_val = n_val - target_pos_val
    if len(neg_patients) > 1:
        target_neg_val = min(target_neg_val, len(neg_patients) - 1)
    else:
        target_neg_val = min(target_neg_val, len(neg_patients))

    if target_pos_val + target_neg_val <= 0:
        target_neg_val = min(1, len(neg_patients))
        if target_neg_val == 0:
            target_pos_val = min(1, len(pos_patients))

    val_pos = rng.choice(pos_patients, size=target_pos_val, replace=False) if target_pos_val > 0 else np.array([], dtype=object)
    val_neg = rng.choice(neg_patients, size=target_neg_val, replace=False) if target_neg_val > 0 else np.array([], dtype=object)
    val_patients = np.concatenate([val_pos, val_neg])
    if len(val_patients) == 0:
        val_patients = rng.choice(all_patients, size=1, replace=False)

    val_set = set(val_patients.tolist())
    train_df = df[~df["Patient_ID"].isin(val_set)].copy()
    val_df = df[df["Patient_ID"].isin(val_set)].copy()

    if train_df.empty or val_df.empty:
        shuffled = all_patients.copy()
        rng.shuffle(shuffled)
        split_idx = max(1, len(shuffled) - 1)
        train_set = set(shuffled[:split_idx].tolist())
        train_df = df[df["Patient_ID"].isin(train_set)].copy()
        val_df = df[~df["Patient_ID"].isin(train_set)].copy()

    train_pos_patients = int((train_df.groupby("Patient_ID")[TARGET_COLUMN].max() > 0).sum()) if not train_df.empty else 0
    val_pos_patients = int((val_df.groupby("Patient_ID")[TARGET_COLUMN].max() > 0).sum()) if not val_df.empty else 0
    return train_df, val_df, train_pos_patients, val_pos_patients


def downsample_negatives(
    x: np.ndarray,
    y: np.ndarray,
    neg_to_pos_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(y) == 0:
        return x, y

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return x, y

    max_neg = int(max(1, neg_to_pos_ratio) * len(pos_idx))
    if len(neg_idx) <= max_neg:
        return x, y

    rng = np.random.default_rng(seed)
    keep_neg = rng.choice(neg_idx, size=max_neg, replace=False)
    keep_idx = np.concatenate([pos_idx, keep_neg])
    rng.shuffle(keep_idx)
    return x[keep_idx], y[keep_idx]


def get_weights(model):
    base = model._module if hasattr(model, "_module") else model
    return [value.detach().cpu().numpy() for _, value in base.state_dict().items()]


def set_weights(model, weights):
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


class NICUFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        fed_cfg: FederatedConfig,
        train_cfg: TrainConfig,
        pos_weight: float,
        client_name: str = "client",
        min_val_positive_sequences: int = 1,
        log_every_n_batches: int = 200,
        val_steps: int | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.fed_cfg = fed_cfg
        self.train_cfg = train_cfg
        self.smpc = SMPCAdapter(enabled=False)
        self.pos_weight = pos_weight
        self.client_name = client_name
        self.min_val_positive_sequences = int(max(1, min_val_positive_sequences))
        self.log_every_n_batches = int(max(1, log_every_n_batches))
        self.val_steps = int(val_steps) if val_steps is not None and val_steps > 0 else None
        self.eval_fallback_max_samples = 10000
        self.focal = BinaryFocalLossWithLogits(
            alpha=self.train_cfg.focal_alpha,
            gamma=self.train_cfg.focal_gamma,
        )

    def get_parameters(self, config):
        params = get_weights(self.model)
        return self.smpc.encrypt_update(params)

    def _collect_eval_outputs(self, max_steps: int | None = None):
        """Run one evaluation pass and return aggregate loss, counts, labels and probs."""
        loss_sum = 0.0
        n = 0
        all_y, all_p = [], []
        with torch.no_grad():
            if max_steps is not None:
                val_iter = iter(self.val_loader)
                for _ in range(max_steps):
                    try:
                        xb, yb = next(val_iter)
                    except StopIteration:
                        val_iter = iter(self.val_loader)
                        xb, yb = next(val_iter)

                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits, _ = self.model(xb)
                    pred = torch.sigmoid(logits)
                    if self.train_cfg.loss_name.lower() == "focal":
                        loss = self.focal(logits, yb)
                    else:
                        loss = weighted_bce_loss(logits, yb, pos_weight=self.pos_weight)
                    loss_sum += loss.item() * len(xb)
                    n += len(xb)
                    all_y.append(yb.detach().cpu().numpy())
                    all_p.append(pred.detach().cpu().numpy())
            else:
                for xb, yb in self.val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits, _ = self.model(xb)
                    pred = torch.sigmoid(logits)
                    if self.train_cfg.loss_name.lower() == "focal":
                        loss = self.focal(logits, yb)
                    else:
                        loss = weighted_bce_loss(logits, yb, pos_weight=self.pos_weight)
                    loss_sum += loss.item() * len(xb)
                    n += len(xb)
                    all_y.append(yb.detach().cpu().numpy())
                    all_p.append(pred.detach().cpu().numpy())

        y_true = np.concatenate(all_y) if all_y else np.array([])
        y_prob = np.concatenate(all_p) if all_p else np.array([])

        y_true = np.asarray(y_true).reshape(-1)
        y_prob = np.asarray(y_prob).reshape(-1)
        finite_mask = np.isfinite(y_true) & np.isfinite(y_prob)
        y_true = y_true[finite_mask]
        y_prob = y_prob[finite_mask]
        return loss_sum, n, y_true, y_prob

    def _collect_eval_outputs_from_indices(self, sample_indices: np.ndarray):
        """Evaluate on a specific subset of validation indices."""
        if len(sample_indices) == 0:
            return 0.0, 0, np.array([]), np.array([])

        x_all, y_all = self.val_loader.dataset.tensors
        idx_tensor = torch.as_tensor(sample_indices, dtype=torch.long)
        x_sub = x_all.index_select(0, idx_tensor)
        y_sub = y_all.index_select(0, idx_tensor)

        loss_sum = 0.0
        n = 0
        all_y, all_p = [], []
        bs = int(self.train_cfg.batch_size)
        with torch.no_grad():
            for start in range(0, len(x_sub), bs):
                xb = x_sub[start : start + bs].to(self.device)
                yb = y_sub[start : start + bs].to(self.device)
                logits, _ = self.model(xb)
                pred = torch.sigmoid(logits)
                if self.train_cfg.loss_name.lower() == "focal":
                    loss = self.focal(logits, yb)
                else:
                    loss = weighted_bce_loss(logits, yb, pos_weight=self.pos_weight)
                loss_sum += loss.item() * len(xb)
                n += len(xb)
                all_y.append(yb.detach().cpu().numpy())
                all_p.append(pred.detach().cpu().numpy())

        y_true = np.concatenate(all_y) if all_y else np.array([])
        y_prob = np.concatenate(all_p) if all_p else np.array([])
        y_true = np.asarray(y_true).reshape(-1)
        y_prob = np.asarray(y_prob).reshape(-1)
        finite_mask = np.isfinite(y_true) & np.isfinite(y_prob)
        y_true = y_true[finite_mask]
        y_prob = y_prob[finite_mask]
        return loss_sum, n, y_true, y_prob

    def fit(self, parameters, config):
        params = self.smpc.decrypt_aggregate(parameters)
        set_weights(self.model, params)
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg.lr)

        train_model = self.model
        train_optimizer = optimizer
        train_loader = self.train_loader
        privacy_engine = None

        if self.fed_cfg.use_dp:
            train_model, train_optimizer, train_loader, privacy_engine = make_private_with_opacus(
                self.model,
                optimizer,
                self.train_loader,
                noise_multiplier=self.fed_cfg.dp_noise_multiplier,
                max_grad_norm=self.fed_cfg.dp_max_grad_norm,
            )

        last_loss = 0.0
        total_batches = max(1, len(train_loader))
        if self.fed_cfg.local_steps is not None and self.fed_cfg.local_steps > 0:
            print(f"[{self.client_name}] fixed-step training enabled | steps={self.fed_cfg.local_steps}")
            data_iter = iter(train_loader)
            for step_idx in range(1, self.fed_cfg.local_steps + 1):
                try:
                    xb, yb = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    xb, yb = next(data_iter)

                xb, yb = xb.to(self.device), yb.to(self.device)
                train_optimizer.zero_grad()
                logits, _ = train_model(xb)
                if self.train_cfg.loss_name.lower() == "focal":
                    loss = self.focal(logits, yb)
                else:
                    loss = weighted_bce_loss(logits, yb, pos_weight=self.pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=self.train_cfg.max_grad_norm)
                train_optimizer.step()
                last_loss = float(loss.item())
                if step_idx == 1 or step_idx % self.log_every_n_batches == 0 or step_idx == self.fed_cfg.local_steps:
                    print(
                        f"[{self.client_name}] step {step_idx}/{self.fed_cfg.local_steps} "
                        f"loss={last_loss:.5f}"
                    )
        else:
            for epoch_idx in range(self.fed_cfg.local_epochs):
                for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    train_optimizer.zero_grad()
                    logits, _ = train_model(xb)
                    if self.train_cfg.loss_name.lower() == "focal":
                        loss = self.focal(logits, yb)
                    else:
                        loss = weighted_bce_loss(logits, yb, pos_weight=self.pos_weight)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=self.train_cfg.max_grad_norm)
                    train_optimizer.step()
                    last_loss = float(loss.item())
                    if batch_idx == 1 or batch_idx % self.log_every_n_batches == 0 or batch_idx == total_batches:
                        print(
                            f"[{self.client_name}] epoch {epoch_idx + 1}/{self.fed_cfg.local_epochs} "
                            f"batch {batch_idx}/{total_batches} loss={last_loss:.5f}"
                        )

        updated = get_weights(train_model)
        metrics = {
            "client_loss": last_loss,
            "train_steps": float(self.fed_cfg.local_steps if self.fed_cfg.local_steps is not None else total_batches * self.fed_cfg.local_epochs),
        }
        if self.fed_cfg.use_dp and privacy_engine is not None:
            metrics["epsilon"] = float(privacy_engine.get_epsilon(delta=self.fed_cfg.delta))

        return self.smpc.encrypt_update(updated), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        params = self.smpc.decrypt_aggregate(parameters)
        set_weights(self.model, params)
        self.model.to(self.device)
        self.model.eval()

        used_full_val = False
        if self.val_steps is not None:
            print(f"[{self.client_name}] fixed-step evaluation enabled | steps={self.val_steps}")
            loss_sum, n, y_true, y_prob = self._collect_eval_outputs(max_steps=self.val_steps)
        else:
            loss_sum, n, y_true, y_prob = self._collect_eval_outputs(max_steps=None)

        pos_count = int(np.sum(y_true == 1))
        neg_count = int(np.sum(y_true == 0))

        # Guard against false EvalSkipped when fixed-step sampling misses positives.
        if self.val_steps is not None and pos_count < self.min_val_positive_sequences:
            # Positive-aware fallback: evaluate all positives plus sampled negatives.
            # This prevents NaN metrics without the very high cost of full-val passes.
            _, y_all = self.val_loader.dataset.tensors
            y_np = y_all.detach().cpu().numpy().reshape(-1)
            pos_idx = np.where(y_np == 1)[0]
            neg_idx = np.where(y_np == 0)[0]

            if len(pos_idx) >= self.min_val_positive_sequences and len(neg_idx) > 0:
                # Keep enough negatives for stable specificity while bounding eval latency.
                target_total = max(len(pos_idx) + 1, len(pos_idx) * 20)
                max_total = int(min(len(y_np), self.eval_fallback_max_samples, target_total))
                n_neg_keep = min(len(neg_idx), max(1, max_total - len(pos_idx)))
                rng = np.random.default_rng(self.train_cfg.random_seed)
                keep_neg = rng.choice(neg_idx, size=n_neg_keep, replace=False)
                eval_idx = np.concatenate([pos_idx, keep_neg])
                rng.shuffle(eval_idx)

                sub_loss_sum, sub_n, sub_y_true, sub_y_prob = self._collect_eval_outputs_from_indices(eval_idx)
                sub_pos = int(np.sum(sub_y_true == 1))
                sub_neg = int(np.sum(sub_y_true == 0))
                print(
                    f"[{self.client_name}] eval fallback | fixed-step positives={pos_count}, "
                    f"using stratified subset positives={sub_pos} negatives={sub_neg}"
                )
                loss_sum, n, y_true, y_prob = sub_loss_sum, sub_n, sub_y_true, sub_y_prob
                pos_count, neg_count = sub_pos, sub_neg
                used_full_val = True

        if pos_count < self.min_val_positive_sequences or neg_count < 1:
            return (
                loss_sum / max(1, n),
                n,
                {
                    "EvalSkipped": 1.0,
                    "ValPositives": float(pos_count),
                    "ValNegatives": float(neg_count),
                    "EvalUsedFullVal": float(used_full_val),
                },
            )

        tuned_threshold, tuned_metrics = select_threshold_for_target_sensitivity(
            y_true,
            y_prob,
            target_sensitivity=self.fed_cfg.target_sensitivity,
        )
        default_metrics = binary_metrics(y_true, y_prob, threshold=self.fed_cfg.default_eval_threshold)
        pr_auc = save_precision_recall_curve(
            y_true,
            y_prob,
            output_path=f"results/pr_curve_{self.client_name}.png",
        )

        eval_metrics = {
            "AUROC": float(default_metrics["AUROC"]),
            "PR_AUC": float(pr_auc),
            "Sensitivity": float(tuned_metrics["Sensitivity"]),
            "Specificity": float(tuned_metrics["Specificity"]),
            "TunedThreshold": float(tuned_threshold),
            "EvalUsedFullVal": float(used_full_val),
        }

        return loss_sum / max(1, n), n, eval_metrics


def main(
    hospital_id: str,
    client_csv: str | None,
    use_dp: bool,
    server_address: str | None,
    dp_noise_multiplier: float | None,
    dp_max_grad_norm: float | None,
    use_mimic: bool = False,
    mimic_dir: str = "mimic-iii-clinical-database-1.4",
    mimic_max_stays: int | None = None,
    mimic_prebuilt_csv: str | None = None,
    seq_len_steps: int = 60,
    client_name: str = "client",
    min_val_positive_patients: int = 1,
    min_val_positive_sequences: int = 1,
    log_every_n_batches: int = 200,
    local_steps: int | None = None,
    val_steps: int | None = None,
    loss_name: str | None = None,
    neg_to_pos_ratio: float | None = None,
):
    data_cfg = DataConfig()
    data_cfg.use_mimic = use_mimic
    data_cfg.mimic_dir = mimic_dir
    data_cfg.mimic_max_stays = mimic_max_stays
    data_cfg.mimic_prebuilt_csv = mimic_prebuilt_csv
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    fed_cfg = FederatedConfig()
    if loss_name:
        train_cfg.loss_name = str(loss_name).lower()
    if neg_to_pos_ratio is not None and neg_to_pos_ratio > 0:
        train_cfg.neg_to_pos_ratio = float(neg_to_pos_ratio)
        fed_cfg.neg_to_pos_ratio = float(neg_to_pos_ratio)
    fed_cfg.use_dp = use_dp
    if server_address:
        fed_cfg.server_address = server_address
    if dp_noise_multiplier is not None:
        fed_cfg.dp_noise_multiplier = dp_noise_multiplier
    if dp_max_grad_norm is not None:
        fed_cfg.dp_max_grad_norm = dp_max_grad_norm
    if local_steps is not None and local_steps > 0:
        fed_cfg.local_steps = int(local_steps)

    if client_csv:
        local_df = pd.read_csv(client_csv)
        local_df["Timestamp"] = pd.to_datetime(local_df["Timestamp"])
        proc_df = preprocess_nicu_data(local_df)
        local_df = proc_df
    else:
        raw_df = load_data(data_cfg)
        proc_df = preprocess_nicu_data(raw_df)
        partitions = partition_by_hospital(proc_df)
        local_df = partitions[hospital_id]

    train_df, val_df, train_pos_patients, val_pos_patients = split_by_patient_with_min_positives(
        local_df,
        test_size=0.2,
        seed=train_cfg.random_seed,
        min_val_positive_patients=min_val_positive_patients,
    )
    x_train, y_train = build_sequences(train_df, seq_len_steps=seq_len_steps)
    x_val, y_val = build_sequences(val_df, seq_len_steps=seq_len_steps)

    x = np.concatenate([x_train, x_val], axis=0) if (len(x_train) + len(x_val)) > 0 else np.empty((0, seq_len_steps, len(MODEL_FEATURE_COLUMNS)), dtype=np.float32)
    y = np.concatenate([y_train, y_val], axis=0) if (len(y_train) + len(y_val)) > 0 else np.empty((0,), dtype=np.float32)

    if len(y) == 0:
        raise RuntimeError(
            "No sequences generated for this client. "
            "Check CSV content, schema alignment, and seq_len_steps."
        )

    print(
        f"[{client_name}] split summary | train_seq={len(y_train)} val_seq={len(y_val)} "
        f"| train_pos_seq={int(np.sum(y_train == 1))} val_pos_seq={int(np.sum(y_val == 1))} "
        f"| train_pos_patients={train_pos_patients} val_pos_patients={val_pos_patients}"
    )

    pre_downsample_prior = float(np.mean(y_train == 1)) if len(y_train) > 0 else 1e-6
    pre_downsample_prior = float(np.clip(pre_downsample_prior, 1e-6, 1.0 - 1e-6))

    # Phase 3: Local Training Downsampling (configurable ratio)
    x_train, y_train = downsample_negatives(
        x_train,
        y_train,
        neg_to_pos_ratio=fed_cfg.neg_to_pos_ratio,
        seed=train_cfg.random_seed,
    )

    train_pos = float(np.sum(y_train == 1))
    train_neg = float(np.sum(y_train == 0))
    dynamic_pos_weight = fed_cfg.pos_class_weight
    if train_pos > 0:
        dynamic_pos_weight = max(1.0, train_neg / train_pos)

    # Log Phase 3 downsampling results and Phase 2 loss configuration
    print(f"[{client_name}] Phase 3: Downsampling (ratio={fed_cfg.neg_to_pos_ratio:.1f}:1) | "
        f"post_ds_seqs: pos={int(train_pos)} neg={int(train_neg)} actual_ratio={train_neg/max(train_pos, 1e-6):.2f}:1")
    print(f"[{client_name}] Phase 2: Loss={train_cfg.loss_name} | pos_weight={dynamic_pos_weight:.2f}")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=train_cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLSTMSepsisModel(
        input_size=x.shape[-1],
        d_model=model_cfg.d_model,
        num_heads=model_cfg.num_heads,
        transformer_layers=model_cfg.transformer_layers,
        lstm_hidden=model_cfg.lstm_hidden,
        lstm_layers=model_cfg.lstm_layers,
        dropout=model_cfg.dropout,
    )
    if fed_cfg.use_dp:
        model = ModuleValidator.fix(model)

    # Phase 1: Initialize output bias exactly once before FL rounds begin.
    if hasattr(model, "initialize_output_bias"):
        model.initialize_output_bias(pre_downsample_prior)
        print(f"[{client_name}] Phase 1: Initialized starting bias | pos_prior={pre_downsample_prior:.6f}")

    client = NICUFlowerClient(
        model,
        train_loader,
        val_loader,
        device,
        fed_cfg,
        train_cfg,
        pos_weight=dynamic_pos_weight,
        client_name=client_name,
        min_val_positive_sequences=min_val_positive_sequences,
        log_every_n_batches=log_every_n_batches,
        val_steps=val_steps,
    )
    fl.client.start_numpy_client(server_address=fed_cfg.server_address, client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hospital_id", type=str, required=True)
    parser.add_argument("--client_csv", type=str, default=None)
    parser.add_argument("--client_name", type=str, default="client")
    parser.add_argument("--use_dp", action="store_true")
    parser.add_argument("--server_address", type=str, default=None)
    parser.add_argument("--dp_noise_multiplier", type=float, default=None)
    parser.add_argument("--dp_max_grad_norm", type=float, default=None)
    parser.add_argument("--seq_len_steps", type=int, default=60)
    parser.add_argument("--min_val_positive_patients", type=int, default=1)
    parser.add_argument("--min_val_positive_sequences", type=int, default=1)
    parser.add_argument("--log_every_n_batches", type=int, default=200)
    parser.add_argument("--local_steps", type=int, default=None)
    parser.add_argument("--val_steps", type=int, default=None)
    parser.add_argument("--loss_name", type=str, default=None, choices=["focal", "bce_weighted"])
    parser.add_argument("--neg_to_pos_ratio", type=float, default=None)
    parser.add_argument("--use_mimic", action="store_true", help="Use real MIMIC-III data.")
    parser.add_argument("--mimic_dir", type=str, default="mimic-iii-clinical-database-1.4")
    parser.add_argument("--mimic_max_stays", type=int, default=None)
    parser.add_argument("--mimic_prebuilt_csv", type=str, default=None)
    args = parser.parse_args()
    main(
        args.hospital_id,
        args.client_csv,
        args.use_dp,
        args.server_address,
        args.dp_noise_multiplier,
        args.dp_max_grad_norm,
        use_mimic=args.use_mimic,
        mimic_dir=args.mimic_dir,
        mimic_max_stays=args.mimic_max_stays,
        mimic_prebuilt_csv=args.mimic_prebuilt_csv,
        seq_len_steps=args.seq_len_steps,
        client_name=args.client_name,
        min_val_positive_patients=args.min_val_positive_patients,
        min_val_positive_sequences=args.min_val_positive_sequences,
        log_every_n_batches=args.log_every_n_batches,
        local_steps=args.local_steps,
        val_steps=args.val_steps,
        loss_name=args.loss_name,
        neg_to_pos_ratio=args.neg_to_pos_ratio,
    )