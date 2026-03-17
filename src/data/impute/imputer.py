from __future__ import annotations

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer


def forward_fill_vitals(
    df: pd.DataFrame,
    vital_cols: list[str],
    group_cols: list[str] | None = None,
    time_col: str = "Timestamp",
) -> pd.DataFrame:
    group_cols = group_cols or ["Hospital_ID", "Patient_ID"]
    out = df.sort_values(group_cols + [time_col]).copy()
    out[vital_cols] = out.groupby(group_cols, sort=False)[vital_cols].ffill()
    return out


def knn_impute_labs(df: pd.DataFrame, lab_cols: list[str], n_neighbors: int = 5) -> pd.DataFrame:
    out = df.copy()
    fallback_defaults = {
        "WBC": 12.0,
        "CRP": 1.0,
        "Platelets": 250.0,
        "Lactate": 1.5,
    }

    for col in lab_cols:
        if col not in out.columns:
            out[col] = fallback_defaults.get(col, 0.0)

    if len(out) > 5000:
        for col in lab_cols:
            series = out[col]
            if series.notna().any():
                median_val = float(series.median())
                out[col] = series.fillna(median_val)
            else:
                out[col] = fallback_defaults.get(col, 0.0)
        return out

    usable_cols = [col for col in lab_cols if out[col].notna().any()]
    empty_cols = [col for col in lab_cols if col not in usable_cols]

    if usable_cols:
        imputer = KNNImputer(n_neighbors=min(n_neighbors, max(1, len(out))))
        imputed = imputer.fit_transform(out[usable_cols])
        out.loc[:, usable_cols] = imputed

    for col in empty_cols:
        out[col] = fallback_defaults.get(col, 0.0)

    return out