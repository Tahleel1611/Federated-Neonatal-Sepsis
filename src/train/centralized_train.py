from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import DataConfig, ModelConfig, TrainConfig
from src.constants import MODEL_FEATURE_COLUMNS, TARGET_COLUMN
from src.data.loaders.synthetic_loader import generate_synthetic_nicu_logs
from src.data.preprocess.pipeline import preprocess_nicu_data
from src.eval.metrics import binary_metrics, save_precision_recall_curve, select_threshold_for_target_sensitivity
from src.models.transformer_lstm import TransformerLSTMSepsisModel


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


def weighted_bce_loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=pred.device, dtype=pred.dtype)
    )
    return criterion(pred, target)


def focal_loss_with_logits(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.80,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt = torch.exp(-bce)
    alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha)
    loss = alpha_t * torch.pow((1.0 - pt), gamma) * bce
    return loss.mean()


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    train_cfg: TrainConfig,
    pos_weight: float,
) -> torch.Tensor:
    if train_cfg.loss_name.lower() == "focal":
        return focal_loss_with_logits(
            pred,
            target,
            alpha=train_cfg.focal_alpha,
            gamma=train_cfg.focal_gamma,
        )
    return weighted_bce_loss(pred, target, pos_weight=pos_weight)


def train_epoch(model, loader, optimizer, device, train_cfg: TrainConfig, pos_weight: float, max_grad_norm: float):
    model.train()
    losses = []
    for batch_idx, (xb, yb) in enumerate(loader, start=1):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = compute_loss(logits, yb, train_cfg=train_cfg, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        losses.append(loss.item())
        if train_cfg.log_every_n_batches > 0 and (
            batch_idx == 1
            or batch_idx % train_cfg.log_every_n_batches == 0
            or batch_idx == train_cfg.steps_per_epoch
        ):
            print(
                f"[train] step {batch_idx}/{train_cfg.steps_per_epoch} "
                f"loss={float(loss.item()):.5f}"
            )
        if batch_idx >= train_cfg.steps_per_epoch:
            break
    return float(np.mean(losses)) if losses else 0.0


def evaluate(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits, _ = model(xb)
            pred = torch.sigmoid(logits)
            all_y.append(yb.numpy())
            all_p.append(pred.cpu().numpy())
    y = np.concatenate(all_y) if all_y else np.array([])
    p = np.concatenate(all_p) if all_p else np.array([])
    return binary_metrics(y, p)


def main(
    data_cfg_override: DataConfig | None = None,
    train_cfg_override: TrainConfig | None = None,
    seq_len_steps: int = 12,
):
    data_cfg = data_cfg_override if data_cfg_override is not None else DataConfig()
    model_cfg = ModelConfig()
    train_cfg = train_cfg_override if train_cfg_override is not None else TrainConfig()
    torch.manual_seed(train_cfg.random_seed)
    np.random.seed(train_cfg.random_seed)

    raw_df = load_data(data_cfg)
    proc_df = preprocess_nicu_data(raw_df)
    x, y = build_sequences(proc_df, seq_len_steps=seq_len_steps)

    use_stratify = (np.unique(y).size > 1) and (np.sum(y == 1) >= 2) and (np.sum(y == 0) >= 2)
    stratify_y = y if use_stratify else None
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=train_cfg.random_seed,
        stratify=stratify_y,
        shuffle=True,
    )

    # Speed guard for large real-world splits: cap validation samples while
    # preserving prevalence for stable metric estimation.
    max_val_samples = 20000
    if len(y_val) > max_val_samples:
        val_stratify = y_val if (np.unique(y_val).size > 1) else None
        _, x_val, _, y_val = train_test_split(
            x_val,
            y_val,
            test_size=max_val_samples,
            random_state=train_cfg.random_seed,
            stratify=val_stratify,
            shuffle=True,
        )

    raw_train_pos = float(np.sum(y_train == 1))
    raw_train_neg = float(np.sum(y_train == 0))
    raw_train_prior = raw_train_pos / max(1.0, raw_train_pos + raw_train_neg)

    x_train_ds, y_train_ds = downsample_negatives(
        x_train,
        y_train,
        neg_to_pos_ratio=train_cfg.neg_to_pos_ratio,
        seed=train_cfg.random_seed,
    )

    train_pos = float(np.sum(y_train_ds == 1))
    train_neg = float(np.sum(y_train_ds == 0))
    dynamic_pos_weight = train_cfg.pos_class_weight
    if raw_train_pos > 0:
        dynamic_pos_weight = max(1.0, raw_train_neg / raw_train_pos)

    print(
        f"[Phase 1] Bias init prior={raw_train_prior:.6f} | "
        f"[Phase 3] Downsampling ratio={train_cfg.neg_to_pos_ratio:.1f}:1 | "
        f"Train/Val split | train_raw={len(y_train)} train_ds={len(y_train_ds)} val={len(y_val)} | "
        f"raw_pos={int(raw_train_pos)} raw_neg={int(raw_train_neg)} | "
        f"ds_pos={int(train_pos)} ds_neg={int(train_neg)} | "
        f"prior={raw_train_prior:.4f} | pos_weight={dynamic_pos_weight:.2f} | "
        f"[Phase 2] loss={train_cfg.loss_name}(alpha={train_cfg.focal_alpha:.2f},gamma={train_cfg.focal_gamma:.1f}) | "
        f"steps_per_epoch={train_cfg.steps_per_epoch}"
    )

    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train_ds), torch.from_numpy(y_train_ds)), batch_size=train_cfg.batch_size, shuffle=True)
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
    ).to(device)
    model.initialize_output_bias(raw_train_prior)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    pr_auc = float("nan")
    best_threshold = train_cfg.default_eval_threshold
    y_val_np = np.array([])
    p_val_np = np.array([])
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, train_cfg.epochs + 1):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train_cfg=train_cfg,
            pos_weight=dynamic_pos_weight,
            max_grad_norm=train_cfg.max_grad_norm,
        )

        model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits, _ = model(xb)
                pred = torch.sigmoid(logits)
                all_y.append(yb.numpy())
                all_p.append(pred.cpu().numpy())

        y_val_np = np.concatenate(all_y) if all_y else np.array([])
        p_val_np = np.concatenate(all_p) if all_p else np.array([])

        best_threshold, tuned_metrics = select_threshold_for_target_sensitivity(
            y_val_np,
            p_val_np,
            target_sensitivity=train_cfg.target_sensitivity,
        )
        default_metrics = binary_metrics(y_val_np, p_val_np, threshold=train_cfg.default_eval_threshold)
        pr_auc = save_precision_recall_curve(y_val_np, p_val_np, output_path="precision_recall_curve.png")

        print(
            " | ".join(
                [
                    f"Epoch {epoch}",
                    f"loss={loss:.4f}",
                    f"AUROC={default_metrics['AUROC']:.4f}",
                    f"PR_AUC={pr_auc:.4f}",
                    f"Sens@0.5={default_metrics['Sensitivity']:.4f}",
                    f"Spec@0.5={default_metrics['Specificity']:.4f}",
                    f"BestThr={best_threshold:.4f}",
                    f"Sens@Best={tuned_metrics['Sensitivity']:.4f}",
                    f"Spec@Best={tuned_metrics['Specificity']:.4f}",
                ]
            )
        )

        history_rows.append(
            {
                "epoch": int(epoch),
                "loss": float(loss),
                "AUROC": float(default_metrics["AUROC"]),
                "PR_AUC": float(pr_auc),
                "Sensitivity_at_0_5": float(default_metrics["Sensitivity"]),
                "Specificity_at_0_5": float(default_metrics["Specificity"]),
                "Tuned_Threshold": float(best_threshold),
                "Sensitivity_tuned": float(tuned_metrics["Sensitivity"]),
                "Specificity_tuned": float(tuned_metrics["Specificity"]),
            }
        )

    print(f"Saved PR curve: precision_recall_curve.png | Final tuned threshold: {best_threshold:.4f} | Final PR_AUC: {pr_auc:.4f}")

    final_default_metrics = binary_metrics(y_val_np, p_val_np, threshold=train_cfg.default_eval_threshold)
    final_tuned_threshold, final_tuned_metrics = select_threshold_for_target_sensitivity(
        y_val_np,
        p_val_np,
        target_sensitivity=train_cfg.target_sensitivity,
    )

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "centralized_model_latest.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": int(x.shape[-1]),
            "d_model": model_cfg.d_model,
            "num_heads": model_cfg.num_heads,
            "transformer_layers": model_cfg.transformer_layers,
            "lstm_hidden": model_cfg.lstm_hidden,
            "lstm_layers": model_cfg.lstm_layers,
            "dropout": model_cfg.dropout,
            "best_threshold": float(final_tuned_threshold),
        },
        checkpoint_path,
    )

    metrics_row = {
        "dataset": "mimic" if data_cfg.use_mimic else "synthetic",
        "epochs": int(train_cfg.epochs),
        "val_samples": int(len(y_val_np)),
        "val_positive": int(np.sum(y_val_np == 1)),
        "AUROC": float(final_default_metrics["AUROC"]),
        "PR_AUC": float(pr_auc),
        "Sensitivity_at_0_5": float(final_default_metrics["Sensitivity"]),
        "Specificity_at_0_5": float(final_default_metrics["Specificity"]),
        "Tuned_Threshold": float(final_tuned_threshold),
        "Sensitivity_tuned": float(final_tuned_metrics["Sensitivity"]),
        "Specificity_tuned": float(final_tuned_metrics["Specificity"]),
    }
    pd.DataFrame([metrics_row]).to_csv(output_dir / "centralized_metrics_latest.csv", index=False)
    if history_rows:
        pd.DataFrame(history_rows).to_csv(output_dir / "centralized_training_history_latest.csv", index=False)
    print(f"Saved checkpoint: {checkpoint_path} | Saved metrics: {output_dir / 'centralized_metrics_latest.csv'}")
    if history_rows:
        print(f"Saved training history: {output_dir / 'centralized_training_history_latest.csv'}")

    if y_val_np.size and p_val_np.size:
        pd.DataFrame({"y_true": y_val_np.astype(int), "y_prob": p_val_np.astype(float)}).to_csv(
            "champion_inference_results.csv", index=False
        )
        pd.DataFrame({"y_true": y_val_np.astype(int), "y_prob": p_val_np.astype(float)}).to_csv(
            "results/champion_inference_results_real.csv", index=False
        )
        print("Saved inference outputs: champion_inference_results.csv and results/champion_inference_results_real.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_mimic", action="store_true", help="Use real MIMIC-III data instead of synthetic.")
    parser.add_argument("--mimic_dir", type=str, default="mimic-iii-clinical-database-1.4")
    parser.add_argument("--mimic_max_stays", type=int, default=None, help="Limit ICU stays (for quick tests).")
    parser.add_argument("--mimic_prebuilt_csv", type=str, default=None, help="Optional path to a pre-extracted MIMIC CSV.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--seq_len_steps", type=int, default=12, help="Sequence length in timesteps.")
    parser.add_argument("--loss_name", type=str, default=None, choices=["focal", "bce_weighted"], help="Loss to use.")
    parser.add_argument("--neg_to_pos_ratio", type=float, default=None, help="Training downsampling ratio (Phase 3).")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Max training batches per epoch (for fast diagnostics).")
    parser.add_argument("--log_every_n_batches", type=int, default=None, help="Training log frequency in batches.")
    args = parser.parse_args()
    # Inject CLI overrides into DataConfig before calling main()
    _data_cfg = DataConfig()
    _data_cfg.use_mimic = args.use_mimic
    _data_cfg.mimic_dir = args.mimic_dir
    _data_cfg.mimic_max_stays = args.mimic_max_stays
    _data_cfg.mimic_prebuilt_csv = args.mimic_prebuilt_csv

    _train_cfg = TrainConfig()
    if args.epochs is not None:
        _train_cfg.epochs = args.epochs
    if args.loss_name is not None:
        _train_cfg.loss_name = str(args.loss_name).lower()
    if args.neg_to_pos_ratio is not None:
        _train_cfg.neg_to_pos_ratio = float(args.neg_to_pos_ratio)
    if args.steps_per_epoch is not None and args.steps_per_epoch > 0:
        _train_cfg.steps_per_epoch = int(args.steps_per_epoch)
    if args.log_every_n_batches is not None and args.log_every_n_batches > 0:
        _train_cfg.log_every_n_batches = int(args.log_every_n_batches)

    main(
        data_cfg_override=_data_cfg,
        train_cfg_override=_train_cfg,
        seq_len_steps=int(args.seq_len_steps),
    )