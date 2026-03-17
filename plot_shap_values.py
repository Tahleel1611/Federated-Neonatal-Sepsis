from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from src.config import ModelConfig
from src.constants import MODEL_FEATURE_COLUMNS, TARGET_COLUMN
from src.data.preprocess.pipeline import preprocess_nicu_data
from src.models.transformer_lstm import TransformerLSTMSepsisModel


def build_sequence_tensor(df: pd.DataFrame, seq_len_steps: int) -> np.ndarray:
    x_list: list[np.ndarray] = []

    ordered = df.sort_values(["Hospital_ID", "Patient_ID", "Timestamp"]).copy()
    ordered["Timestamp"] = pd.to_datetime(ordered["Timestamp"])

    for _, group in ordered.groupby(["Hospital_ID", "Patient_ID"], sort=False):
        values = group[MODEL_FEATURE_COLUMNS + [TARGET_COLUMN]].to_numpy(dtype=float)
        for idx in range(seq_len_steps, len(values)):
            x_list.append(values[idx - seq_len_steps : idx, :-1])

    if not x_list:
        raise RuntimeError("No sequences built. Check dataset content and seq_len_steps.")

    return np.asarray(x_list, dtype=np.float32)


class ProbabilityWrapper(torch.nn.Module):
    def __init__(self, base_model: TransformerLSTMSepsisModel) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.base_model(x)
        return torch.sigmoid(logits).unsqueeze(-1)


def load_model(checkpoint_path: Path, input_size: int, device: torch.device) -> tuple[torch.nn.Module, int]:
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_cfg = ModelConfig()
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        seq_len_steps = int(ckpt.get("seq_len_steps", 60))
        model = TransformerLSTMSepsisModel(
            input_size=int(ckpt.get("input_size", input_size)),
            d_model=int(ckpt.get("d_model", model_cfg.d_model)),
            num_heads=int(ckpt.get("num_heads", model_cfg.num_heads)),
            transformer_layers=int(ckpt.get("transformer_layers", model_cfg.transformer_layers)),
            lstm_hidden=int(ckpt.get("lstm_hidden", model_cfg.lstm_hidden)),
            lstm_layers=int(ckpt.get("lstm_layers", model_cfg.lstm_layers)),
            dropout=float(ckpt.get("dropout", model_cfg.dropout)),
        )
    else:
        state_dict = ckpt
        seq_len_steps = 60
        model = TransformerLSTMSepsisModel(input_size=input_size)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return ProbabilityWrapper(model).to(device).eval(), seq_len_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SHAP values for federated Transformer-LSTM sequences.")
    parser.add_argument("--checkpoint_path", type=str, default="results/federated_global_model_final.pt")
    parser.add_argument("--dataset_csv", type=str, default="results/datasets/mimic_nicu_H1_200.csv")
    parser.add_argument("--seq_len_steps", type=int, default=60)
    parser.add_argument("--background_size", type=int, default=32)
    parser.add_argument("--explain_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_png", type=str, default="results/final_shap_summary.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset_csv = Path(args.dataset_csv)
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")

    raw_df = pd.read_csv(dataset_csv)
    proc_df = preprocess_nicu_data(raw_df)

    seq_len_steps = int(args.seq_len_steps)
    x = build_sequence_tensor(proc_df, seq_len_steps=seq_len_steps)

    model, ckpt_seq_len = load_model(checkpoint_path, input_size=x.shape[-1], device=device)
    if seq_len_steps != ckpt_seq_len:
        print(f"Warning: CLI seq_len={seq_len_steps}, checkpoint seq_len={ckpt_seq_len}. Using CLI-built sequences.")

    n_samples = x.shape[0]
    bg_n = int(max(8, min(args.background_size, n_samples)))
    ex_n = int(max(8, min(args.explain_size, n_samples - bg_n if n_samples > bg_n else n_samples)))

    rng = np.random.default_rng(int(args.seed))
    perm = rng.permutation(n_samples)
    bg_idx = perm[:bg_n]
    ex_idx = perm[bg_n : bg_n + ex_n] if n_samples > bg_n else perm[:ex_n]

    background = torch.tensor(x[bg_idx], dtype=torch.float32, device=device)
    explain = torch.tensor(x[ex_idx], dtype=torch.float32, device=device)

    print(f"Computing SHAP values on {explain.shape[0]} sequences with {background.shape[0]} background sequences...")
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(explain)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)
    explain_np = explain.detach().cpu().numpy()

    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]
    if shap_values.ndim != 3:
        raise RuntimeError(f"Unexpected SHAP output shape: {shap_values.shape}. Expected (N, T, F).")

    shap_values_2d = np.mean(shap_values, axis=1)
    explain_2d = np.mean(explain_np, axis=1)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_2d,
        explain_2d,
        feature_names=MODEL_FEATURE_COLUMNS,
        show=False,
    )
    plt.title("SHAP Global Feature Importance (Time-Averaged)")
    plt.tight_layout()

    out_png = Path(args.output_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    mean_abs = np.mean(np.abs(shap_values_2d), axis=0)
    top_idx = np.argsort(-mean_abs)[:10]
    print("Top 10 features by mean(|SHAP|):")
    for rank, idx in enumerate(top_idx, start=1):
        print(f"{rank:2d}. {MODEL_FEATURE_COLUMNS[idx]}: {mean_abs[idx]:.6f}")

    print(f"Saved SHAP summary plot: {out_png}")


if __name__ == "__main__":
    main()
