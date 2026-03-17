from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.config import ModelConfig, TrainConfig
from src.constants import MODEL_FEATURE_COLUMNS, TARGET_COLUMN
from src.data.preprocess.pipeline import preprocess_nicu_data
from src.models.transformer_lstm import TransformerLSTMSepsisModel


def build_sequences_with_metadata(df: pd.DataFrame, seq_len_steps: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    meta_rows: list[dict[str, object]] = []

    ordered = df.sort_values(["Hospital_ID", "Patient_ID", "Timestamp"]).copy()
    ordered["Timestamp"] = pd.to_datetime(ordered["Timestamp"])

    for _, group in ordered.groupby(["Hospital_ID", "Patient_ID"], sort=False):
        values = group[MODEL_FEATURE_COLUMNS + [TARGET_COLUMN]].to_numpy(dtype=float)
        hospital_ids = group["Hospital_ID"].to_numpy()
        patient_ids = group["Patient_ID"].to_numpy()
        timestamps = group["Timestamp"].to_numpy()

        for idx in range(seq_len_steps, len(values)):
            x_list.append(values[idx - seq_len_steps : idx, :-1])
            y_list.append(values[idx, -1])
            meta_rows.append(
                {
                    "Hospital_ID": hospital_ids[idx],
                    "Patient_ID": patient_ids[idx],
                    "Timestamp": timestamps[idx],
                }
            )

    x = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    return x, y, meta


def load_and_prepare_combined_dataframe(mimic_csv: Path, pic_csv: Path) -> pd.DataFrame:
    mimic_df = pd.read_csv(mimic_csv)
    pic_df = pd.read_csv(pic_csv)

    combined = pd.concat([mimic_df, pic_df], axis=0, ignore_index=True)
    combined["Timestamp"] = pd.to_datetime(combined["Timestamp"])
    return preprocess_nicu_data(combined)


def split_holdout(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    test_size: float,
    random_seed: int,
    max_test_samples: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    all_idx = np.arange(len(y))
    use_stratify = (np.unique(y).size > 1) and (np.sum(y == 1) >= 2) and (np.sum(y == 0) >= 2)
    stratify_y = y if use_stratify else None

    _, test_idx, _, y_test = train_test_split(
        all_idx,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_y,
        shuffle=True,
    )

    if len(y_test) > max_test_samples:
        test_stratify = y_test if np.unique(y_test).size > 1 else None
        _, test_idx, _, y_test = train_test_split(
            test_idx,
            y_test,
            test_size=max_test_samples,
            random_state=random_seed,
            stratify=test_stratify,
            shuffle=True,
        )

    x_test = x[test_idx]
    y_test = y_test.astype(np.float32)
    meta_test = meta.iloc[test_idx].reset_index(drop=True).copy()
    return x_test, y_test, meta_test


def load_model_from_checkpoint(checkpoint_path: Path, input_size: int) -> tuple[TransformerLSTMSepsisModel, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = ModelConfig()

    model = TransformerLSTMSepsisModel(
        input_size=int(checkpoint.get("input_size", input_size)),
        d_model=int(checkpoint.get("d_model", model_cfg.d_model)),
        num_heads=int(checkpoint.get("num_heads", model_cfg.num_heads)),
        transformer_layers=int(checkpoint.get("transformer_layers", model_cfg.transformer_layers)),
        lstm_hidden=int(checkpoint.get("lstm_hidden", model_cfg.lstm_hidden)),
        lstm_layers=int(checkpoint.get("lstm_layers", model_cfg.lstm_layers)),
        dropout=float(checkpoint.get("dropout", model_cfg.dropout)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    seq_len_steps = int(checkpoint.get("seq_len_steps", 60))
    return model, seq_len_steps


def run_inference(
    model: TransformerLSTMSepsisModel,
    x_test: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = DataLoader(TensorDataset(torch.from_numpy(x_test)), batch_size=batch_size, shuffle=False)

    probs: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits, _ = model(xb)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(prob)

    if not probs:
        return np.array([], dtype=np.float32)
    return np.concatenate(probs).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with federated global checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, default="results/federated_global_model_final.pt")
    parser.add_argument("--mimic_csv", type=str, default="results/datasets/mimic_nicu_dataset_50.csv")
    parser.add_argument("--pic_csv", type=str, default="results/datasets/pic_nicu_aligned.csv")
    parser.add_argument("--seq_len_steps", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=20000)
    parser.add_argument("--output_csv", type=str, default="champion_inference_results.csv")
    parser.add_argument(
        "--output_with_meta_csv",
        type=str,
        default="results/champion_inference_results_federated_with_meta.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    train_cfg = TrainConfig()
    batch_size = int(args.batch_size) if args.batch_size is not None else int(train_cfg.batch_size)

    temp_model, ckpt_seq_len = load_model_from_checkpoint(checkpoint_path, input_size=len(MODEL_FEATURE_COLUMNS))
    seq_len_steps = int(args.seq_len_steps) if args.seq_len_steps is not None else int(ckpt_seq_len)

    combined_df = load_and_prepare_combined_dataframe(Path(args.mimic_csv), Path(args.pic_csv))
    x, y, meta = build_sequences_with_metadata(combined_df, seq_len_steps=seq_len_steps)
    if len(y) == 0:
        raise RuntimeError("No evaluation sequences were built. Check dataset content and seq_len_steps.")

    x_test, y_test, meta_test = split_holdout(
        x=x,
        y=y,
        meta=meta,
        test_size=float(args.test_size),
        random_seed=int(args.random_seed),
        max_test_samples=int(args.max_test_samples),
    )

    y_prob = run_inference(temp_model, x_test, batch_size=batch_size)
    if len(y_prob) != len(y_test):
        raise RuntimeError(
            f"Prediction length mismatch: got {len(y_prob)} probabilities for {len(y_test)} labels."
        )

    output_df = pd.DataFrame(
        {
            "y_true": y_test.astype(int),
            "y_prob": y_prob.astype(float),
        }
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    with_meta = meta_test.copy()
    with_meta["y_true"] = y_test.astype(int)
    with_meta["y_prob"] = y_prob.astype(float)
    with_meta_path = Path(args.output_with_meta_csv)
    with_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with_meta.to_csv(with_meta_path, index=False)

    print(
        "Saved federated inference outputs: "
        f"{output_path} (rows={len(output_df)}) and {with_meta_path}"
    )


if __name__ == "__main__":
    main()
