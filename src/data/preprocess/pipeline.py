from __future__ import annotations

import pandas as pd

from src.constants import HRV_COLUMNS, LAB_AGE_COLUMNS, LAB_COLUMNS, LAB_MASK_COLUMNS, MODEL_FEATURE_COLUMNS, STATIC_COLUMNS, VITAL_COLUMNS
from src.data.features.hrv import compute_hrv_sliding_window
from src.data.impute.imputer import forward_fill_vitals, knn_impute_labs


def add_lab_mask_and_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["Hospital_ID", "Patient_ID"]

    for lab_col, mask_col, recency_col in zip(LAB_COLUMNS, LAB_MASK_COLUMNS, LAB_AGE_COLUMNS):
        out[mask_col] = out[lab_col].notna().astype(float)
        out[recency_col] = 0.0

        grouped = out.groupby(group_cols, sort=False)
        for _, row_idx in grouped.groups.items():
            patient_idx = list(row_idx)
            group_df = out.loc[patient_idx, ["Timestamp", lab_col]]

            measured_ts = group_df["Timestamp"].where(group_df[lab_col].notna()).ffill()
            recency_h = (group_df["Timestamp"] - measured_ts).dt.total_seconds() / 3600.0

            from_start_h = (group_df["Timestamp"] - group_df["Timestamp"].iloc[0]).dt.total_seconds() / 3600.0
            out.loc[patient_idx, recency_col] = recency_h.fillna(from_start_h).astype(float).to_numpy()

    return out


def preprocess_nicu_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Timestamp"] = pd.to_datetime(out["Timestamp"])
    out = out.sort_values(["Hospital_ID", "Patient_ID", "Timestamp"])

    out = add_lab_mask_and_recency_features(out)
    out = forward_fill_vitals(out, VITAL_COLUMNS)
    out[VITAL_COLUMNS] = (
        out.groupby(["Hospital_ID", "Patient_ID"], sort=False)[VITAL_COLUMNS].bfill().values
    )
    out = knn_impute_labs(out, LAB_COLUMNS)
    out = compute_hrv_sliding_window(out)
    out[HRV_COLUMNS] = out[HRV_COLUMNS].fillna(0.0)
    out[LAB_MASK_COLUMNS] = out[LAB_MASK_COLUMNS].fillna(0.0)
    out[LAB_AGE_COLUMNS] = out[LAB_AGE_COLUMNS].fillna(0.0)

    feature_cols = MODEL_FEATURE_COLUMNS
    out[feature_cols] = out[feature_cols].fillna(out[feature_cols].median(numeric_only=True))
    out[feature_cols] = out[feature_cols].replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    return out