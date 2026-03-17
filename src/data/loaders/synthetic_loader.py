from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import DataConfig
from src.constants import KEY_COLUMNS, LAB_COLUMNS, STATIC_COLUMNS, TARGET_COLUMN, VITAL_COLUMNS


def _patient_static(rng: np.random.Generator) -> dict:
    return {
        "Birth_Weight": float(np.clip(rng.normal(2.6, 0.6), 0.8, 4.8)),
        "Gestational_Age": float(np.clip(rng.normal(36.0, 2.8), 26, 42)),
    }


def generate_synthetic_nicu_logs(config: DataConfig, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start_time = datetime(2026, 1, 1)
    points = (config.observation_hours * 60) // config.time_step_minutes

    for hospital_idx in range(1, config.n_hospitals + 1):
        hospital_id = f"H{hospital_idx}"
        for patient_idx in range(1, config.n_patients_per_hospital + 1):
            patient_id = f"{hospital_id}_P{patient_idx:03d}"
            static = _patient_static(rng)

            latent_risk = rng.uniform(0, 1)
            sepsis_start_point = rng.integers(points // 2, points) if latent_risk > 0.5 else points + 100

            for t in range(points):
                ts = start_time + timedelta(minutes=t * config.time_step_minutes)
                trend = max(0, t - sepsis_start_point + points // 4)

                hr = rng.normal(135 + 0.15 * trend, 9)
                spo2 = rng.normal(96 - 0.06 * trend, 2)
                rr = rng.normal(42 + 0.08 * trend, 6)
                temp = rng.normal(36.8 + 0.02 * trend, 0.3)

                wbc = rng.normal(12 + 0.03 * trend, 2.5)
                crp = rng.normal(1.2 + 0.015 * trend, 0.9)
                platelets = rng.normal(280 - 0.4 * trend, 40)
                lactate = rng.normal(1.4 + 0.006 * trend, 0.4)

                future_window_start = t
                future_window_end = t + (config.prediction_horizon_hours * 60) // config.time_step_minutes
                label = int(sepsis_start_point >= future_window_start and sepsis_start_point <= future_window_end)

                row = {
                    "Hospital_ID": hospital_id,
                    "Patient_ID": patient_id,
                    "Timestamp": ts,
                    "HR": float(np.clip(hr, 80, 220)),
                    "SpO2": float(np.clip(spo2, 70, 100)),
                    "RR": float(np.clip(rr, 15, 90)),
                    "Temp": float(np.clip(temp, 34.5, 40.5)),
                    "WBC": float(np.clip(wbc, 1, 40)),
                    "CRP": float(np.clip(crp, 0, 35)),
                    "Platelets": float(np.clip(platelets, 30, 600)),
                    "Lactate": float(np.clip(lactate, 0.2, 10)),
                    TARGET_COLUMN: label,
                }
                row.update(static)
                rows.append(row)

    df = pd.DataFrame(rows)

    for col in VITAL_COLUMNS + LAB_COLUMNS:
        miss_mask = rng.uniform(0, 1, size=len(df)) < (0.06 if col in VITAL_COLUMNS else 0.12)
        df.loc[miss_mask, col] = np.nan

    return df[KEY_COLUMNS + VITAL_COLUMNS + LAB_COLUMNS + STATIC_COLUMNS + [TARGET_COLUMN]]