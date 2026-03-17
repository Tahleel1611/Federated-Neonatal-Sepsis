from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import binary_metrics


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    timestamps: np.ndarray,
    diagnosis_time_by_patient: dict[str, float],
    patient_ids: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    metrics = binary_metrics(y_true, y_prob, threshold=threshold)

    lead_times = []
    triggered = y_prob >= threshold
    df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "timestamp": timestamps,
            "triggered": triggered,
        }
    )

    for patient_id, group in df.groupby("patient_id", sort=False):
        trigger_rows = group[group["triggered"]]
        if trigger_rows.empty or patient_id not in diagnosis_time_by_patient:
            continue
        first_trigger = float(trigger_rows["timestamp"].min())
        diagnosis = float(diagnosis_time_by_patient[patient_id])
        lead_hours = (diagnosis - first_trigger) / 3600.0
        lead_times.append(lead_hours)

    metrics["LeadTimeHoursMean"] = float(np.mean(lead_times)) if lead_times else float("nan")
    metrics["LeadTimeHoursMedian"] = float(np.median(lead_times)) if lead_times else float("nan")
    return metrics