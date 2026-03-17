from __future__ import annotations

import numpy as np
import pandas as pd


def _sdnn(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    return float(np.std(values, ddof=1))


def _rmssd(values: np.ndarray) -> float:
    if len(values) < 3:
        return np.nan
    diffs = np.diff(values)
    return float(np.sqrt(np.mean(np.square(diffs))))


def compute_hrv_sliding_window(
    df: pd.DataFrame,
    patient_col: str = "Patient_ID",
    time_col: str = "Timestamp",
    hr_col: str = "HR",
    window: str = "60min",
) -> pd.DataFrame:
    ordered = df.sort_values([patient_col, time_col]).copy()
    ordered[time_col] = pd.to_datetime(ordered[time_col])

    out = []
    for patient_id, group in ordered.groupby(patient_col, sort=False):
        temp = group.set_index(time_col)
        hr = temp[hr_col].astype(float)
        temp["HRV_SDNN"] = hr.rolling(window=window, min_periods=3).apply(lambda x: _sdnn(x.to_numpy()), raw=False)
        temp["HRV_RMSSD"] = hr.rolling(window=window, min_periods=3).apply(lambda x: _rmssd(x.to_numpy()), raw=False)
        temp[patient_col] = patient_id
        out.append(temp.reset_index())

    return pd.concat(out, ignore_index=True)