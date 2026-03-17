from __future__ import annotations

import argparse
import numpy as np
import shap
import torch
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

from src.config import DataConfig, ModelConfig, TrainConfig
from src.constants import MODEL_FEATURE_COLUMNS, TARGET_COLUMN
from src.data.loaders.synthetic_loader import generate_synthetic_nicu_logs
from src.data.preprocess.pipeline import preprocess_nicu_data
from src.models.transformer_lstm import TransformerLSTMSepsisModel


def build_tabular_features(df):
    feature_cols = MODEL_FEATURE_COLUMNS
    grouped = df.groupby("Patient_ID", sort=False).tail(1)
    x = grouped[feature_cols].to_numpy(dtype=np.float32)
    y = grouped[TARGET_COLUMN].to_numpy(dtype=np.float32)
    return x, y, feature_cols


def load_data(data_cfg: DataConfig) -> pd.DataFrame:
    if data_cfg.use_mimic:
        if data_cfg.mimic_prebuilt_csv and Path(data_cfg.mimic_prebuilt_csv).exists():
            return pd.read_csv(data_cfg.mimic_prebuilt_csv)
        from src.data.loaders.mimic_loader import build_mimic_nicu_dataset
        return build_mimic_nicu_dataset(
            mimic_dir=data_cfg.mimic_dir,
            prediction_horizon_h=data_cfg.prediction_horizon_hours,
            max_stays=data_cfg.mimic_max_stays,
        )
    return generate_synthetic_nicu_logs(data_cfg, seed=TrainConfig().random_seed)


def main(
    use_mimic: bool = False,
    mimic_prebuilt_csv: str | None = None,
    checkpoint_path: str = "results/centralized_model_latest.pt",
    output_png: str = "results/shap_high_risk_summary_real.png",
):
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    data_cfg.use_mimic = use_mimic
    data_cfg.mimic_prebuilt_csv = mimic_prebuilt_csv

    raw_df = load_data(data_cfg)
    proc_df = preprocess_nicu_data(raw_df)
    x, y, feature_cols = build_tabular_features(proc_df)

    baseline = np.repeat(x[:, None, :], repeats=12, axis=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLSTMSepsisModel(
        input_size=baseline.shape[-1],
        d_model=model_cfg.d_model,
        num_heads=model_cfg.num_heads,
        transformer_layers=model_cfg.transformer_layers,
        lstm_hidden=model_cfg.lstm_hidden,
        lstm_layers=model_cfg.lstm_layers,
        dropout=model_cfg.dropout,
    ).to(device)

    ckpt_file = Path(checkpoint_path)
    if ckpt_file.exists():
        ckpt = torch.load(ckpt_file, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {ckpt_file}")
    else:
        print(f"Checkpoint not found ({ckpt_file}); using current model weights.")

    model.eval()

    high_risk_idx = np.where(y == 1)[0][:50]
    if len(high_risk_idx) == 0:
        high_risk_idx = np.arange(min(50, len(x)))
    sample = x[high_risk_idx]

    def predict_fn(x_tab: np.ndarray) -> np.ndarray:
        seq = np.repeat(x_tab[:, None, :], repeats=12, axis=1)
        with torch.no_grad():
            logits, _ = model(torch.tensor(seq, dtype=torch.float32, device=device))
            probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy()

    background = sample[: min(20, len(sample))]
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(sample[: min(30, len(sample))], nsamples=100)

    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, sample[: min(30, len(sample))], feature_names=feature_cols, show=False)
    plt.title("FedNeo-Guard SHAP Feature Importance (High-Risk Alerts)")
    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=150)
    print(f"Saved SHAP summary plot: {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_mimic", action="store_true")
    parser.add_argument("--mimic_prebuilt_csv", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default="results/centralized_model_latest.pt")
    parser.add_argument("--output_png", type=str, default="results/shap_high_risk_summary_real.png")
    args = parser.parse_args()
    main(
        use_mimic=args.use_mimic,
        mimic_prebuilt_csv=args.mimic_prebuilt_csv,
        checkpoint_path=args.checkpoint_path,
        output_png=args.output_png,
    )