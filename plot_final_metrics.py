from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def main() -> None:
    csv_path = Path("results/champion_inference_results_federated_final.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"y_true", "y_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    y_true = df["y_true"].to_numpy(dtype=int)
    y_prob = df["y_prob"].to_numpy(dtype=float)

    if y_true.size == 0:
        raise ValueError("Input CSV is empty.")

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Plot 1: ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Federated Global Model (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Chance")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "final_roc_curve.png", dpi=300)
    plt.close()

    # Plot 2: Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    prevalence = float(np.mean(y_true))

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color="purple", lw=2, label=f"Federated Global Model (PR-AUC = {pr_auc:.4f})")
    plt.plot([0, 1], [prevalence, prevalence], color="navy", lw=2, linestyle="--", label=f"Baseline Prevalence ({prevalence:.4f})")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "final_pr_curve.png", dpi=300)
    plt.close()

    # Plot 3: Confusion Matrix at target sensitivity
    target_sensitivity = 0.8316
    threshold_idx = int(np.argmin(np.abs(tpr - target_sensitivity)))
    optimal_threshold = float(roc_thresholds[threshold_idx])

    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_optimal, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Healthy", "Predicted Sepsis"],
        yticklabels=["Actual Healthy", "Actual Sepsis"],
    )
    plt.title(f"Confusion Matrix (Threshold = {optimal_threshold:.4f})")
    plt.tight_layout()
    plt.savefig(out_dir / "final_confusion_matrix.png", dpi=300)
    plt.close()

    print("Plots saved to results/: final_roc_curve.png, final_pr_curve.png, final_confusion_matrix.png")
    print(f"ROC AUC: {roc_auc:.6f}")
    print(f"PR AUC: {pr_auc:.6f}")
    print(f"Target sensitivity: {target_sensitivity:.4f} | Selected threshold: {optimal_threshold:.6f}")


if __name__ == "__main__":
    main()
