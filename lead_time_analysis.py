from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from src.config import TrainConfig
from src.constants import TARGET_COLUMN
from src.data.preprocess.pipeline import preprocess_nicu_data


def build_sequence_metadata(df: pd.DataFrame, seq_len_steps: int = 12) -> tuple[np.ndarray, pd.DataFrame]:
    labels: list[int] = []
    meta_rows: list[dict[str, object]] = []

    ordered = df.sort_values(["Hospital_ID", "Patient_ID", "Timestamp"]).copy()
    ordered["Timestamp"] = pd.to_datetime(ordered["Timestamp"])

    for _, group in ordered.groupby("Patient_ID", sort=False):
        group = group.reset_index(drop=True)
        for idx in range(seq_len_steps, len(group)):
            row = group.iloc[idx]
            labels.append(int(row[TARGET_COLUMN]))
            meta_rows.append(
                {
                    "Hospital_ID": row["Hospital_ID"],
                    "Patient_ID": row["Patient_ID"],
                    "Timestamp": row["Timestamp"],
                    "y_true": int(row[TARGET_COLUMN]),
                }
            )

    return np.asarray(labels, dtype=np.int64), pd.DataFrame(meta_rows)


def reconstruct_validation_metadata(
    dataset_csv: Path,
    seq_len_steps: int,
    test_size: float,
    random_seed: int,
    max_val_samples: int,
) -> pd.DataFrame:
    raw_df = pd.read_csv(dataset_csv)
    proc_df = preprocess_nicu_data(raw_df)
    y, meta = build_sequence_metadata(proc_df, seq_len_steps=seq_len_steps)

    all_idx = np.arange(len(y))
    use_stratify = (np.unique(y).size > 1) and (np.sum(y == 1) >= 2) and (np.sum(y == 0) >= 2)
    stratify_y = y if use_stratify else None

    _, val_idx, _, y_val = train_test_split(
        all_idx,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_y,
        shuffle=True,
    )

    if len(y_val) > max_val_samples:
        val_stratify = y_val if np.unique(y_val).size > 1 else None
        _, val_idx, _, y_val = train_test_split(
            val_idx,
            y_val,
            test_size=max_val_samples,
            random_state=random_seed,
            stratify=val_stratify,
            shuffle=True,
        )

    val_meta = meta.iloc[val_idx].reset_index(drop=True).copy()
    val_meta["y_true"] = y_val.astype(int)
    return val_meta


def attach_prediction_metadata(inference_csv: Path, val_meta: pd.DataFrame) -> pd.DataFrame:
    pred_df = pd.read_csv(inference_csv)
    required = {"y_true", "y_prob"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in inference CSV: {sorted(missing)}")

    pred_df = pred_df[["y_true", "y_prob"]].copy()
    pred_df["y_true"] = pred_df["y_true"].astype(int)
    pred_df["y_prob"] = pred_df["y_prob"].astype(float)

    if len(pred_df) != len(val_meta):
        raise ValueError(
            f"Prediction rows ({len(pred_df)}) do not match reconstructed validation rows ({len(val_meta)})."
        )

    if not np.array_equal(pred_df["y_true"].to_numpy(dtype=int), val_meta["y_true"].to_numpy(dtype=int)):
        raise ValueError(
            "Reconstructed validation labels do not align with champion inference labels. "
            "Check dataset path, seed, sequence length, or split parameters."
        )

    merged = val_meta.copy()
    merged["y_prob"] = pred_df["y_prob"].to_numpy(dtype=float)
    return merged


def compute_proxy_onset_times(df: pd.DataFrame, prediction_horizon_hours: float) -> dict[str, pd.Timestamp]:
    onset_by_patient: dict[str, pd.Timestamp] = {}
    for patient_id, group in df.groupby("Patient_ID", sort=False):
        positive_rows = group[group["y_true"] == 1].sort_values("Timestamp")
        if positive_rows.empty:
            continue
        first_positive_ts = pd.to_datetime(positive_rows["Timestamp"].iloc[0])
        onset_by_patient[patient_id] = first_positive_ts + pd.Timedelta(hours=prediction_horizon_hours)
    return onset_by_patient


def compute_lead_time_summary(
    df: pd.DataFrame,
    threshold: float,
    prediction_horizon_hours: float,
) -> pd.DataFrame:
    onset_by_patient = compute_proxy_onset_times(df, prediction_horizon_hours=prediction_horizon_hours)
    lead_rows: list[dict[str, object]] = []

    for patient_id, onset_ts in onset_by_patient.items():
        patient_df = df[df["Patient_ID"] == patient_id].copy()
        patient_df["Timestamp"] = pd.to_datetime(patient_df["Timestamp"])
        patient_df = patient_df.sort_values("Timestamp")

        window_start = onset_ts - pd.Timedelta(hours=prediction_horizon_hours)
        window_df = patient_df[(patient_df["Timestamp"] >= window_start) & (patient_df["Timestamp"] <= onset_ts)].copy()
        trigger_df = window_df[window_df["y_prob"] >= threshold].sort_values("Timestamp")

        caught = not trigger_df.empty
        trigger_ts = pd.NaT if not caught else pd.to_datetime(trigger_df["Timestamp"].iloc[0])
        lead_hours = float((onset_ts - trigger_ts).total_seconds() / 3600.0) if caught else np.nan

        lead_rows.append(
            {
                "Patient_ID": patient_id,
                "Proxy_Onset_Timestamp": onset_ts,
                "Window_Start_Timestamp": window_start,
                "Triggered": bool(caught),
                "Trigger_Timestamp": trigger_ts,
                "Lead_Time_Hours": lead_hours,
                "Max_Prob_In_Window": float(window_df["y_prob"].max()) if not window_df.empty else np.nan,
                "Min_Prob_In_Window": float(window_df["y_prob"].min()) if not window_df.empty else np.nan,
                "Positive_Window_Rows": int(len(window_df)),
            }
        )

    summary_df = pd.DataFrame(lead_rows).sort_values(["Triggered", "Lead_Time_Hours"], ascending=[False, False])
    return summary_df


def save_lead_time_histogram(summary_df: pd.DataFrame, threshold: float, output_png: Path) -> None:
    caught = summary_df[summary_df["Triggered"] & summary_df["Lead_Time_Hours"].notna()].copy()

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5.5))

    if caught.empty:
        plt.text(0.5, 0.5, "No sepsis cases crossed the selected threshold in the pre-onset window.", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        sns.histplot(caught["Lead_Time_Hours"], bins=12, kde=True, color="#4C72B0")
        median_lead = float(caught["Lead_Time_Hours"].median())
        plt.axvline(median_lead, linestyle="--", color="black", linewidth=1.5, label=f"Median = {median_lead:.2f} h")
        plt.legend()

    plt.title(f"Lead-Time Distribution for Threshold {threshold:.4f}")
    plt.xlabel("Lead Time Before Proxy Sepsis Onset (hours)")
    plt.ylabel("Number of Sepsis Patients")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()


def print_summary(summary_df: pd.DataFrame, threshold: float, prediction_horizon_hours: float) -> None:
    total_sepsis_patients = int(len(summary_df))
    caught_df = summary_df[summary_df["Triggered"] & summary_df["Lead_Time_Hours"].notna()].copy()
    caught_count = int(len(caught_df))
    catch_rate = caught_count / total_sepsis_patients if total_sepsis_patients > 0 else float("nan")
    median_lead = float(caught_df["Lead_Time_Hours"].median()) if caught_count > 0 else float("nan")
    mean_lead = float(caught_df["Lead_Time_Hours"].mean()) if caught_count > 0 else float("nan")

    print("\n=== Lead-Time Analysis Summary ===")
    print(f"Threshold: {threshold:.4f}")
    print(f"Prediction horizon: {prediction_horizon_hours:.1f} hours")
    print(f"Sepsis patients in validation set: {total_sepsis_patients}")
    print(f"Caught before proxy onset: {caught_count} ({catch_rate:.2%})")
    print(f"Median lead time: {median_lead:.2f} h")
    print(f"Mean lead time: {mean_lead:.2f} h")
    if caught_count > 0:
        q25, q75 = np.quantile(caught_df["Lead_Time_Hours"], [0.25, 0.75])
        print(f"Lead-time IQR: {q25:.2f} h to {q75:.2f} h")

    print(
        "\nNote: MIMIC lead time uses a proxy onset reconstructed from the dataset label design "
        f"(first positive label timestamp + {prediction_horizon_hours:.1f} h), because a true clinical onset timestamp is not stored in the exported CSV."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate patient-level lead time before proxy sepsis onset.")
    parser.add_argument("--inference_csv", type=str, default="champion_inference_results.csv")
    parser.add_argument("--dataset_csv", type=str, default="results/mimic_nicu_dataset_50.csv")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--prediction_horizon_hours", type=float, default=6.0)
    parser.add_argument("--seq_len_steps", type=int, default=12)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_val_samples", type=int, default=20000)
    parser.add_argument("--output_csv", type=str, default="results/lead_time_patient_summary.csv")
    parser.add_argument("--output_png", type=str, default="results/lead_time_histogram.png")
    parser.add_argument("--output_merged_csv", type=str, default="results/champion_inference_with_metadata.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    val_meta = reconstruct_validation_metadata(
        dataset_csv=Path(args.dataset_csv),
        seq_len_steps=args.seq_len_steps,
        test_size=args.test_size,
        random_seed=args.random_seed,
        max_val_samples=args.max_val_samples,
    )
    merged = attach_prediction_metadata(Path(args.inference_csv), val_meta)
    merged.to_csv(args.output_merged_csv, index=False)

    summary_df = compute_lead_time_summary(
        merged,
        threshold=args.threshold,
        prediction_horizon_hours=args.prediction_horizon_hours,
    )
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_csv, index=False)

    save_lead_time_histogram(summary_df, threshold=args.threshold, output_png=Path(args.output_png))
    print_summary(summary_df, threshold=args.threshold, prediction_horizon_hours=args.prediction_horizon_hours)
    print(f"Saved merged inference metadata: {args.output_merged_csv}")
    print(f"Saved patient lead-time summary: {args.output_csv}")
    print(f"Saved lead-time histogram: {args.output_png}")


if __name__ == "__main__":
    main()