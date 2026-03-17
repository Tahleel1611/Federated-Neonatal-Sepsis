from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    finite_mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[finite_mask]
    y_prob = y_prob[finite_mask]
    if y_true.size == 0:
        return {
            "AUROC": float("nan"),
            "Sensitivity": float("nan"),
            "Specificity": float("nan"),
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
        }

    y_true = np.clip(y_true, 0.0, 1.0).astype(int)
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)

    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    if np.unique(y_true).size < 2:
        auroc = float("nan")
    else:
        try:
            auroc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auroc = float("nan")

    return {
        "AUROC": auroc,
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def _confusion_metrics_for_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return {
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def compute_lead_time_hours(
    trigger_times: np.ndarray,
    diagnosis_times: np.ndarray,
) -> np.ndarray:
    return (diagnosis_times - trigger_times) / 3600.0


def select_threshold_for_target_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float = 0.80,
) -> tuple[float, dict]:
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    finite_mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[finite_mask]
    y_prob = np.nan_to_num(y_prob[finite_mask], nan=0.5, posinf=1.0, neginf=0.0)

    if len(y_true) == 0:
        return 0.5, {"Sensitivity": float("nan"), "Specificity": float("nan")}

    candidate_thresholds = np.unique(np.clip(y_prob, 0.0, 1.0))
    if len(candidate_thresholds) == 0:
        return 0.5, {"Sensitivity": float("nan"), "Specificity": float("nan")}

    # Cap threshold grid size to keep FL evaluation latency bounded.
    max_thresholds = 512
    if candidate_thresholds.size > max_thresholds:
        quantiles = np.linspace(0.0, 1.0, max_thresholds)
        candidate_thresholds = np.unique(np.quantile(candidate_thresholds, quantiles))

    if np.unique(np.clip(y_true, 0.0, 1.0).astype(int)).size < 2:
        auroc = float("nan")
    else:
        try:
            auroc = float(roc_auc_score(np.clip(y_true, 0.0, 1.0).astype(int), y_prob))
        except ValueError:
            auroc = float("nan")

    best_threshold = 0.5
    best_metrics = _confusion_metrics_for_threshold(y_true, y_prob, threshold=best_threshold)
    best_metrics["AUROC"] = auroc
    best_specificity = -1.0
    fallback_candidates = []

    for threshold in candidate_thresholds:
        metrics = _confusion_metrics_for_threshold(y_true, y_prob, threshold=float(threshold))
        metrics["AUROC"] = auroc
        sensitivity = metrics["Sensitivity"]
        specificity = metrics["Specificity"]
        fallback_candidates.append((abs(sensitivity - target_sensitivity), float(threshold), metrics))
        if sensitivity >= target_sensitivity and specificity > best_specificity:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_specificity = float(specificity)

    if best_specificity < 0:
        fallback_candidates.sort(key=lambda x: x[0])
        _, best_threshold, best_metrics = fallback_candidates[0]

    return best_threshold, best_metrics


def save_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    finite_mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[finite_mask]
    y_prob = np.nan_to_num(y_prob[finite_mask], nan=0.5, posinf=1.0, neginf=0.0)

    if len(y_true) == 0:
        return float("nan")

    if np.unique(y_true).size < 2:
        return float("nan")

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall, precision))

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Sepsis)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return pr_auc