from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def _safe_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy=strategy,
    )
    return np.asarray(prob_true), np.asarray(prob_pred)


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"y_true", "y_prob_raw", "y_prob_calibrated"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns in input CSV: "
            f"{sorted(missing)}. Expected calibrated file from calibrate_probabilities.py"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reliability curves for raw vs isotonic-calibrated probabilities.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="results/champion_inference_results_calibrated.csv",
        help="CSV with columns: y_true, y_prob_raw, y_prob_calibrated",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="results/calibration_reliability_plot.png",
        help="Output PNG path for two-panel reliability figure",
    )
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default="results/calibration_plot_metrics.csv",
        help="Output CSV with Brier and calibration summary",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=15,
        help="Number of bins for calibration curves",
    )
    parser.add_argument(
        "--hist_bins",
        type=int,
        default=40,
        help="Number of bins for prediction density histogram",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="quantile",
        choices=["quantile", "uniform"],
        help="Binning strategy for calibration_curve",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Sepsis Risk Calibration: Raw vs Isotonic",
        help="Top-panel title",
    )
    parser.add_argument(
        "--hist_log_scale",
        action="store_true",
        help="Use log scale on histogram y-axis (recommended for extreme imbalance)",
    )
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)
    _validate_columns(df)

    work = df[["y_true", "y_prob_raw", "y_prob_calibrated"]].dropna().copy()
    work["y_true"] = work["y_true"].astype(int)
    work["y_prob_raw"] = work["y_prob_raw"].astype(float).clip(0.0, 1.0)
    work["y_prob_calibrated"] = work["y_prob_calibrated"].astype(float).clip(0.0, 1.0)

    y_true = work["y_true"].to_numpy()
    p_raw = work["y_prob_raw"].to_numpy()
    p_cal = work["y_prob_calibrated"].to_numpy()

    if len(np.unique(y_true)) < 2:
        raise ValueError("Need both classes present to build reliability curves.")

    brier_raw = float(brier_score_loss(y_true, p_raw))
    brier_cal = float(brier_score_loss(y_true, p_cal))

    frac_pos_raw, mean_pred_raw = _safe_calibration_curve(
        y_true,
        p_raw,
        n_bins=int(args.n_bins),
        strategy=str(args.strategy),
    )
    frac_pos_cal, mean_pred_cal = _safe_calibration_curve(
        y_true,
        p_cal,
        n_bins=int(args.n_bins),
        strategy=str(args.strategy),
    )

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
    )

    # Top panel: reliability curves
    ax_top.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Ideal calibration")
    ax_top.plot(
        mean_pred_raw,
        frac_pos_raw,
        marker="o",
        linewidth=2,
        label=f"Raw probabilities (Brier={brier_raw:.6f})",
    )
    ax_top.plot(
        mean_pred_cal,
        frac_pos_cal,
        marker="o",
        linewidth=2,
        label=f"Isotonic calibrated (Brier={brier_cal:.6f})",
    )
    ax_top.set_ylabel("Observed event frequency")
    ax_top.set_title(args.title)
    ax_top.set_xlim(0.0, 1.0)
    ax_top.set_ylim(0.0, 1.0)
    ax_top.grid(alpha=0.25)
    ax_top.legend(loc="best")

    # Bottom panel: prediction density histogram
    hist_bins = np.linspace(0.0, 1.0, int(args.hist_bins) + 1)
    ax_bottom.hist(
        p_raw,
        bins=hist_bins,
        alpha=0.45,
        label="Raw probabilities",
        color="#1f77b4",
    )
    ax_bottom.hist(
        p_cal,
        bins=hist_bins,
        alpha=0.45,
        label="Isotonic calibrated",
        color="#ff7f0e",
    )
    if args.hist_log_scale:
        ax_bottom.set_yscale("log")
    ax_bottom.set_xlabel("Predicted probability")
    ax_bottom.set_ylabel("Count")
    ax_bottom.grid(alpha=0.25)
    ax_bottom.legend(loc="best")

    fig.tight_layout()

    out_png = Path(args.output_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    metrics = {
        "n_samples": int(len(work)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "n_bins": int(args.n_bins),
        "hist_bins": int(args.hist_bins),
        "strategy": str(args.strategy),
        "hist_log_scale": bool(args.hist_log_scale),
        "brier_raw": brier_raw,
        "brier_calibrated": brier_cal,
        "brier_delta_cal_minus_raw": float(brier_cal - brier_raw),
    }

    out_metrics = Path(args.metrics_csv)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out_metrics, index=False)

    print("Calibration plotting complete.")
    print(f"Saved two-panel reliability plot: {out_png}")
    print(f"Saved plot metrics: {out_metrics}")
    print(f"Brier raw: {brier_raw:.6f}")
    print(f"Brier calibrated: {brier_cal:.6f}")


if __name__ == "__main__":
    main()
