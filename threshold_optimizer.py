from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


INPUT_CSV = Path("champion_inference_results.csv")
THRESHOLD_DYNAMICS_PNG = Path("threshold_dynamics.png")
CONFUSION_MATRICES_PNG = Path("optimized_confusion_matrices.png")


def compute_confusion_components(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f2 = (5.0 * precision * sensitivity) / (4.0 * precision + sensitivity) if (4.0 * precision + sensitivity) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f2": float(f2),
    }


def find_youden_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    return float(thresholds[best_idx]), float(youden_j[best_idx]), thresholds, youden_j, tpr


def find_f2_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

    if len(pr_thresholds) == 0:
        default_metrics = compute_confusion_components(y_true, y_prob, 0.5)
        return 0.5, float(default_metrics["f2"]), np.array([0.5]), np.array([float(default_metrics["f2"])])

    precision_aligned = precision[:-1]
    recall_aligned = recall[:-1]
    denominator = (4.0 * precision_aligned) + recall_aligned
    f2_scores = np.divide(
        5.0 * precision_aligned * recall_aligned,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    best_idx = int(np.argmax(f2_scores))
    return float(pr_thresholds[best_idx]), float(f2_scores[best_idx]), pr_thresholds, f2_scores


def annotate_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    total = cm.sum()
    if total == 0:
        return np.array([["0\n(0.00%)" for _ in range(cm.shape[1])] for _ in range(cm.shape[0])], dtype=object)

    labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = (cm[i, j] / total) * 100.0
            labels[i, j] = f"{cm[i, j]}\n({pct:.2f}%)"
    return labels


def save_threshold_dynamics_plot(
    youden_thresholds: np.ndarray,
    youden_scores: np.ndarray,
    f2_thresholds: np.ndarray,
    f2_scores: np.ndarray,
    optimal_youden_threshold: float,
    optimal_f2_threshold: float,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(youden_thresholds, youden_scores, label="Youden's J", linewidth=2)
    plt.plot(f2_thresholds, f2_scores, label="F2-Score", linewidth=2)
    plt.axvline(optimal_youden_threshold, linestyle="--", linewidth=1.5, label=f"Best Youden: {optimal_youden_threshold:.4f}")
    plt.axvline(optimal_f2_threshold, linestyle=":", linewidth=1.5, label=f"Best F2: {optimal_f2_threshold:.4f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Dynamics: Youden's J and F2-Score")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_optimized_confusion_matrices(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    youden_threshold: float,
    f2_threshold: float,
    output_path: Path,
) -> None:
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    settings = [
        ("Youden Optimal", youden_threshold, axes[0]),
        ("F2 Optimal", f2_threshold, axes[1]),
    ]

    for title, threshold, ax in settings:
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        annotations = annotate_confusion_matrix(cm)
        sns.heatmap(
            cm,
            annot=annotations,
            fmt="",
            cmap="Blues",
            cbar=False,
            linewidths=0.6,
            linecolor="white",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            ax=ax,
        )
        ax.set_title(f"{title}\nThreshold = {threshold:.4f}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def print_summary(default_metrics: dict[str, float | int], youden_metrics: dict[str, float | int], f2_metrics: dict[str, float | int]) -> None:
    print("\n=== Threshold Optimization Summary ===")
    print(f"Input file: {INPUT_CSV}")
    print()
    print(f"{'Setting':<14} {'Threshold':>10} {'Sensitivity':>13} {'Specificity':>13} {'Precision':>11} {'F2-Score':>10}")
    print("-" * 76)

    rows = [
        ("Default 0.5", default_metrics),
        ("Youden", youden_metrics),
        ("F2", f2_metrics),
    ]

    for label, metrics in rows:
        print(
            f"{label:<14} "
            f"{float(metrics['threshold']):>10.4f} "
            f"{float(metrics['sensitivity']):>13.4f} "
            f"{float(metrics['specificity']):>13.4f} "
            f"{float(metrics['precision']):>11.4f} "
            f"{float(metrics['f2']):>10.4f}"
        )

    print("\nConfusion counts (TP, TN, FP, FN):")
    for label, metrics in rows:
        print(
            f"- {label:<11}: "
            f"TP={int(metrics['tp'])}, TN={int(metrics['tn'])}, FP={int(metrics['fp'])}, FN={int(metrics['fn'])}"
        )


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Could not find {INPUT_CSV}. Run this script from the project root.")

    df = pd.read_csv(INPUT_CSV)
    required_columns = {"y_true", "y_prob"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    eval_df = df[list(required_columns)].dropna().copy()
    if eval_df.empty:
        raise ValueError("No valid rows found after dropping NaN values in y_true/y_prob.")

    y_true = eval_df["y_true"].astype(int).to_numpy()
    y_prob = eval_df["y_prob"].astype(float).to_numpy()

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary values 0 or 1.")

    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("y_prob values must be in [0.0, 1.0].")

    youden_threshold, _, roc_thresholds, youden_scores, _ = find_youden_optimal_threshold(y_true, y_prob)
    f2_threshold, _, pr_thresholds, f2_scores = find_f2_optimal_threshold(y_true, y_prob)

    default_metrics = compute_confusion_components(y_true, y_prob, 0.5)
    youden_metrics = compute_confusion_components(y_true, y_prob, youden_threshold)
    f2_metrics = compute_confusion_components(y_true, y_prob, f2_threshold)

    save_threshold_dynamics_plot(
        youden_thresholds=roc_thresholds,
        youden_scores=youden_scores,
        f2_thresholds=pr_thresholds,
        f2_scores=f2_scores,
        optimal_youden_threshold=youden_threshold,
        optimal_f2_threshold=f2_threshold,
        output_path=THRESHOLD_DYNAMICS_PNG,
    )

    save_optimized_confusion_matrices(
        y_true=y_true,
        y_prob=y_prob,
        youden_threshold=youden_threshold,
        f2_threshold=f2_threshold,
        output_path=CONFUSION_MATRICES_PNG,
    )

    print_summary(default_metrics, youden_metrics, f2_metrics)
    print(f"\nSaved: {THRESHOLD_DYNAMICS_PNG} and {CONFUSION_MATRICES_PNG}")


if __name__ == "__main__":
    main()