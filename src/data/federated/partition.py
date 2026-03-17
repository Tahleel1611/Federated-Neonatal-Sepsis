from __future__ import annotations

import pandas as pd


def partition_by_hospital(df: pd.DataFrame, hospital_col: str = "Hospital_ID") -> dict[str, pd.DataFrame]:
    partitions: dict[str, pd.DataFrame] = {}
    for hospital_id, group in df.groupby(hospital_col, sort=False):
        partitions[str(hospital_id)] = group.reset_index(drop=True)
    return partitions