from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config import ModelConfig, TrainConfig
from src.constants import MODEL_FEATURE_COLUMNS
from src.fl.federated_clients import ClientBundle, get_federated_dataloaders
from src.models.transformer_lstm import TransformerLSTMSepsisModel


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.80, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        loss = alpha_t * torch.pow(1.0 - pt, self.gamma) * bce

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


# Phase 2: BCEWithLogitsLoss wrapper for weighted loss (handles imbalance directly)
class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight]),
            reduction=reduction
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        # Update pos_weight tensor device and dtype to match input
        self.criterion.pos_weight = self.criterion.pos_weight.to(logits.device).type(logits.dtype)
        return self.criterion(logits, target)


def create_model(input_size: int) -> TransformerLSTMSepsisModel:
    model_cfg = ModelConfig()
    return TransformerLSTMSepsisModel(
        input_size=input_size,
        d_model=model_cfg.d_model,
        num_heads=model_cfg.num_heads,
        transformer_layers=model_cfg.transformer_layers,
        lstm_hidden=model_cfg.lstm_hidden,
        lstm_layers=model_cfg.lstm_layers,
        dropout=model_cfg.dropout,
    )


def train_client_local(
    model: nn.Module,
    bundle: ClientBundle,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    criterion: nn.Module,
    max_grad_norm: float,
) -> tuple[nn.Module, dict[str, float]]:
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history: list[float] = []
    for _ in range(epochs):
        for xb, yb in bundle.dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            loss_history.append(float(loss.item()))

    metrics = {
        "train_loss": float(np.mean(loss_history)) if loss_history else 0.0,
        "num_samples": float(bundle.num_samples),
        "positive_prior": float(bundle.positive_prior),
    }
    return model, metrics


def federated_average(
    global_model: nn.Module,
    client_models: dict[str, nn.Module],
    client_sample_counts: dict[str, int],
) -> nn.Module:
    global_state = global_model.state_dict()
    total_samples = max(1, int(sum(client_sample_counts.values())))

    for key in global_state.keys():
        reference_tensor = client_models[next(iter(client_models))].state_dict()[key]
        if not reference_tensor.is_floating_point():
            global_state[key] = reference_tensor.clone()
            continue

        aggregated = None
        for client_name, local_model in client_models.items():
            client_weight = client_sample_counts[client_name] / total_samples
            local_tensor = local_model.state_dict()[key].detach().float() * client_weight
            aggregated = local_tensor if aggregated is None else aggregated + local_tensor
        global_state[key] = aggregated.type_as(reference_tensor)

    global_model.load_state_dict(global_state)
    return global_model


def pooled_positive_prior(client_bundles: dict[str, ClientBundle]) -> float:
    total_samples = sum(bundle.num_samples for bundle in client_bundles.values())
    if total_samples == 0:
        return 0.5
    positive = sum(bundle.positive_prior * bundle.num_samples for bundle in client_bundles.values())
    return float(positive / total_samples)


def run_federated_training(
    mimic_csv: str | Path,
    pic_csv: str | Path,
    rounds: int = 10,
    local_epochs: int = 3,
    batch_size: int = 32,
    seq_len_steps: int = 60,
    learning_rate: float = 3e-4,
    save_path: str | Path | None = None,
) -> nn.Module:
    train_cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_loaders, client_weights, client_bundles = get_federated_dataloaders(
        mimic_csv_path=mimic_csv,
        pic_csv_path=pic_csv,
        batch_size=batch_size,
        seq_len_steps=seq_len_steps,
        shuffle=True,
        print_sanity_summary=True,
    )
    del client_loaders, client_weights

    global_model = create_model(input_size=len(MODEL_FEATURE_COLUMNS)).to(device)
    global_model.initialize_output_bias(pooled_positive_prior(client_bundles))
    
    # Phase 2: Support both focal loss and weighted BCEWithLogitsLoss
    if train_cfg.loss_name.lower() == "focal":
        criterion = BinaryFocalLossWithLogits(alpha=train_cfg.focal_alpha, gamma=train_cfg.focal_gamma)
        print(f"[FedNeo-Guard] Using BinaryFocalLossWithLogits | alpha={train_cfg.focal_alpha} gamma={train_cfg.focal_gamma}")
    else:
        # For "bce_weighted", we'll dynamically set pos_weight later based on client data
        criterion = WeightedBCEWithLogitsLoss(pos_weight=train_cfg.pos_class_weight)
        print(f"[FedNeo-Guard] Using WeightedBCEWithLogitsLoss | default_pos_weight={train_cfg.pos_class_weight}")

    for round_num in range(rounds):
        print(f"\n{'=' * 12} Federated Round {round_num + 1}/{rounds} {'=' * 12}")
        client_models: dict[str, nn.Module] = {}
        client_sample_counts = {name: bundle.num_samples for name, bundle in client_bundles.items()}

        for client_name, bundle in client_bundles.items():
            print(f"--> Broadcasting to {client_name} | local_sequences={bundle.num_samples}")
            local_model = copy.deepcopy(global_model)
            trained_model, local_metrics = train_client_local(
                model=local_model,
                bundle=bundle,
                epochs=local_epochs,
                learning_rate=learning_rate,
                device=device,
                criterion=criterion,
                max_grad_norm=train_cfg.max_grad_norm,
            )
            client_models[client_name] = trained_model.cpu()
            print(
                f"    local_loss={local_metrics['train_loss']:.4f} | "
                f"samples={int(local_metrics['num_samples'])} | prior={local_metrics['positive_prior']:.4f}"
            )

        print("Aggregating weights on Central Server (FedAvg)...")
        global_model = federated_average(global_model.cpu(), client_models, client_sample_counts).to(device)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": global_model.cpu().state_dict(),
                "input_size": len(MODEL_FEATURE_COLUMNS),
                "seq_len_steps": int(seq_len_steps),
                "rounds": int(rounds),
                "local_epochs": int(local_epochs),
                "learning_rate": float(learning_rate),
            },
            save_path,
        )
        print(f"Saved federated checkpoint to {save_path}")

    print("\nFederated Training Complete!")
    return global_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight 2-client FedAvg training.")
    parser.add_argument("--mimic_csv", type=str, required=True)
    parser.add_argument("--pic_csv", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len_steps", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--save_path", type=str, default="results/fedavg_model_latest.pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_federated_training(
        mimic_csv=args.mimic_csv,
        pic_csv=args.pic_csv,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        seq_len_steps=args.seq_len_steps,
        learning_rate=args.learning_rate,
        save_path=args.save_path,
    )