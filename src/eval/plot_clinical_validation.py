from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"y_true", "y_prob"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df[["y_true", "y_prob"]].copy()
    out["y_true"] = out["y_true"].astype(int)
    out["y_prob"] = out["y_prob"].astype(float)

    invalid_y = ~out["y_true"].isin([0, 1])
    if invalid_y.any():
        raise ValueError("Column 'y_true' must contain only 0 or 1 values.")

    out["y_prob"] = out["y_prob"].clip(0.0, 1.0)
    return out


def plot_clinical_validation_grid(
    input_csv: str = "champion_inference_results.csv",
    output_png: str = "clinical_validation_grid.png",
    threshold: float = 0.5,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("colorblind")

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    df = pd.read_csv(input_csv)
    df = _validate_input(df)

    y_true = df["y_true"].to_numpy()
    y_prob = df["y_prob"].to_numpy()
    y_pred = (y_prob >= threshold).astype(int)

    prevalence = float(np.mean(y_true))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Neonatal Sepsis Model: Clinical Validation Panel", fontsize=16, y=1.02)

    ax_roc = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    ax_roc.plot(fpr, tpr, linewidth=2.5, label=f"ROC (AUROC = {auroc:.3f})")
    ax_roc.fill_between(fpr, tpr, alpha=0.2)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="gray", label="Random Chance")
    ax_roc.set_title("Receiver Operating Characteristic")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.legend(loc="lower right")

    ax_pr = axes[0, 1]
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall, precision)
    ax_pr.plot(recall, precision, linewidth=2.5, label=f"PR (AUPRC = {auprc:.3f})")
    ax_pr.fill_between(recall, precision, alpha=0.2)
    ax_pr.axhline(prevalence, linestyle="--", linewidth=1.5, color="gray", label=f"Baseline Prevalence = {prevalence:.3f}")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    ax_pr.legend(loc="lower left")

    ax_cm = axes[1, 0]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_percent = (cm / cm.sum()) * 100.0 if cm.sum() > 0 else np.zeros_like(cm, dtype=float)
    annotations = np.array(
        [
            [f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ]
    )

    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        xticklabels=["Healthy (0)", "Sepsis (1)"],
        yticklabels=["Healthy (0)", "Sepsis (1)"],
        ax=ax_cm,
    )
    ax_cm.set_title(f"Clinical Confusion Matrix (Threshold = {threshold:.2f})")
    ax_cm.set_xlabel("Predicted Sepsis")
    ax_cm.set_ylabel("Actual Sepsis")

    ax_dist = axes[1, 1]
    sns.histplot(
        data=df,
        x="y_prob",
        hue="y_true",
        bins=25,
        stat="density",
        common_norm=False,
        alpha=0.45,
        kde=True,
        palette=["#4C72B0", "#DD8452"],
        ax=ax_dist,
    )
    ax_dist.axvline(threshold, linestyle="--", linewidth=1.5, color="black", label=f"Threshold = {threshold:.2f}")
    ax_dist.set_title("Distribution of Predicted Sepsis Probability")
    ax_dist.set_xlabel("Predicted Probability (y_prob)")
    ax_dist.set_ylabel("Density")
    handles, labels = ax_dist.get_legend_handles_labels()
    mapped_labels = ["Healthy (Actual 0)", "Sepsis (Actual 1)"]
    if len(handles) >= 2:
        ax_dist.legend(handles[:2] + [handles[-1]] if len(handles) > 2 else handles, mapped_labels + [f"Threshold = {threshold:.2f}"] if len(handles) > 2 else mapped_labels)

    fig.tight_layout()
    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved clinical validation grid: {out_path}")
    print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | Prevalence: {prevalence:.4f} | Threshold: {threshold:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-ready 2x2 clinical validation plots.")
    parser.add_argument("--input_csv", type=str, default="champion_inference_results.csv")
    parser.add_argument("--output_png", type=str, default="clinical_validation_grid.png")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_clinical_validation_grid(
        input_csv=args.input_csv,
        output_png=args.output_png,
        threshold=args.threshold,
    )