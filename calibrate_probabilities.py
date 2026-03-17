from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc isotonic calibration for sepsis probabilities.")
    parser.add_argument("--input_csv", type=str, default="champion_inference_results.csv")
    parser.add_argument("--output_csv", type=str, default="results/champion_inference_results_calibrated.csv")
    parser.add_argument("--report_csv", type=str, default="results/calibration_report.csv")
    parser.add_argument("--calibration_frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inp = Path(args.input_csv)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    df = pd.read_csv(inp)
    required = {"y_true", "y_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df[["y_true", "y_prob"]].dropna().copy()
    work["y_true"] = work["y_true"].astype(int)
    work["y_prob"] = work["y_prob"].astype(float).clip(0.0, 1.0)

    y = work["y_true"].to_numpy()
    p = work["y_prob"].to_numpy()

    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes present for calibration/evaluation.")

    idx = np.arange(len(work))
    idx_cal, idx_eval = train_test_split(
        idx,
        test_size=max(0.1, 1.0 - float(args.calibration_frac)),
        random_state=int(args.seed),
        stratify=y,
        shuffle=True,
    )

    y_cal, p_cal = y[idx_cal], p[idx_cal]
    y_eval, p_eval = y[idx_eval], p[idx_eval]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    p_eval_cal = iso.transform(p_eval)
    p_all_cal = iso.transform(p)

    metrics = {
        "split_size_cal": int(len(idx_cal)),
        "split_size_eval": int(len(idx_eval)),
        "auroc_raw_eval": float(roc_auc_score(y_eval, p_eval)),
        "auroc_cal_eval": float(roc_auc_score(y_eval, p_eval_cal)),
        "auprc_raw_eval": float(average_precision_score(y_eval, p_eval)),
        "auprc_cal_eval": float(average_precision_score(y_eval, p_eval_cal)),
        "brier_raw_eval": float(brier_score_loss(y_eval, p_eval)),
        "brier_cal_eval": float(brier_score_loss(y_eval, p_eval_cal)),
        "logloss_raw_eval": float(log_loss(y_eval, np.clip(p_eval, 1e-9, 1 - 1e-9))),
        "logloss_cal_eval": float(log_loss(y_eval, np.clip(p_eval_cal, 1e-9, 1 - 1e-9))),
        "ece_raw_eval": float(ece_score(y_eval, p_eval)),
        "ece_cal_eval": float(ece_score(y_eval, p_eval_cal)),
    }

    out_df = work.copy()
    out_df["y_prob_raw"] = out_df["y_prob"]
    out_df["y_prob_calibrated"] = p_all_cal
    out_df = out_df[["y_true", "y_prob_raw", "y_prob_calibrated"]]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    report_path = Path(args.report_csv)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(report_path, index=False)

    print("Calibration complete.")
    print(f"Saved calibrated probabilities: {out_path}")
    print(f"Saved calibration report: {report_path}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
