from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MIMIC_BASE_COLS = [
    "Hospital_ID",
    "Patient_ID",
    "Timestamp",
    "HR",
    "SpO2",
    "RR",
    "Temp",
    "WBC",
    "CRP",
    "Platelets",
    "Lactate",
    "Birth_Weight",
    "Gestational_Age",
    "Sepsis_onset_next_6h",
]

LABS = ["WBC", "CRP", "Platelets", "Lactate"]

PIC_ITEMIDS = {
    "HR": [1003, 1002],
    "SpO2": [1006],
    "RR": [1004],
    "Temp": [1001],
    "WBC": [5141],
    "CRP": [5626, 5821],
    "Platelets": [5129],
    "Lactate": [5227],
}


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def select_event_available_subjects(
    base_dir: Path,
    candidate_subject_ids: set[int],
    chart_itemids: list[int],
    lab_itemids: list[int],
    max_subjects: int,
) -> list[int]:
    chart_subjects: set[int] = set()
    for chunk in pd.read_csv(base_dir / "CHARTEVENTS.csv", usecols=["SUBJECT_ID", "ITEMID"], chunksize=1_000_000):
        chunk = chunk[chunk["SUBJECT_ID"].isin(candidate_subject_ids)]
        chunk = chunk[chunk["ITEMID"].isin(chart_itemids)]
        if not chunk.empty:
            chart_subjects.update(chunk["SUBJECT_ID"].astype(int).tolist())

    lab_subjects: set[int] = set()
    for chunk in pd.read_csv(base_dir / "LABEVENTS.csv", usecols=["SUBJECT_ID", "ITEMID"], chunksize=1_000_000):
        chunk = chunk[chunk["SUBJECT_ID"].isin(candidate_subject_ids)]
        chunk = chunk[chunk["ITEMID"].isin(lab_itemids)]
        if not chunk.empty:
            lab_subjects.update(chunk["SUBJECT_ID"].astype(int).tolist())

    selected = sorted(chart_subjects & lab_subjects)
    if not selected:
        selected = sorted(chart_subjects)
    if not selected:
        selected = sorted(candidate_subject_ids)
    return selected[:max_subjects]


def build_pediatric_cohort(base_dir: Path, max_subjects: int | None = None) -> pd.DataFrame:
    icu = pd.read_csv(
        base_dir / "ICUSTAYS.csv",
        usecols=["SUBJECT_ID", "HADM_ID", "INTIME", "OUTTIME", "FIRST_CAREUNIT", "LAST_CAREUNIT"],
    )
    patients = pd.read_csv(base_dir / "PATIENTS.csv", usecols=["SUBJECT_ID", "DOB"])
    admissions = pd.read_csv(base_dir / "ADMISSIONS.csv", usecols=["SUBJECT_ID", "HADM_ID", "DIAGNOSIS", "ICD10_CODE_CN"])
    # All ICD codes per admission (secondary diagnoses included)
    diagnoses_icd = pd.read_csv(base_dir / "DIAGNOSES_ICD.csv", usecols=["SUBJECT_ID", "HADM_ID", "ICD10_CODE_CN"])

    icu["INTIME"] = pd.to_datetime(icu["INTIME"], errors="coerce")
    patients["DOB"] = pd.to_datetime(patients["DOB"], errors="coerce")

    cohort = icu.merge(patients, on="SUBJECT_ID", how="left").merge(admissions, on=["SUBJECT_ID", "HADM_ID"], how="left")
    age_years = (cohort["INTIME"] - cohort["DOB"]).dt.total_seconds() / (365.25 * 24 * 3600)
    cohort = cohort[(age_years >= 0) & (age_years <= 18)].copy()

    careunit_text = (
        cohort["FIRST_CAREUNIT"].fillna("").astype(str) + " " + cohort["LAST_CAREUNIT"].fillna("").astype(str)
    ).str.lower()
    icu_mask = careunit_text.str.contains("nicu|neonat|picu|pediatric|children|icu", regex=True)
    if icu_mask.any():
        cohort = cohort[icu_mask].copy()

    # Build sepsis HADM set from: primary ICD (ADMISSIONS) + all secondary ICDs (DIAGNOSES_ICD)
    # DIAGNOSIS text is in Chinese ("败血" = septicemia) so we match both scripts
    _icd_pattern = r"a40|a41|p36|r65"
    _primary_sepsis = cohort["ICD10_CODE_CN"].fillna("").str.contains(_icd_pattern, case=False, regex=True)
    _text_sepsis = cohort["DIAGNOSIS"].fillna("").str.contains(
        r"sepsis|败血|脓毒", case=False, regex=True
    )
    _secondary_sepsis_hadms = set(
        diagnoses_icd[
            diagnoses_icd["ICD10_CODE_CN"].fillna("").str.contains(_icd_pattern, case=False, regex=True)
        ]["HADM_ID"].tolist()
    )
    sepsis_mask = (
        _primary_sepsis
        | _text_sepsis
        | cohort["HADM_ID"].isin(_secondary_sepsis_hadms)
    )
    cohort["Sepsis_onset_next_6h"] = sepsis_mask.astype(float)

    cohort = cohort.drop_duplicates(subset=["SUBJECT_ID", "HADM_ID"]).copy()

    return cohort[["SUBJECT_ID", "HADM_ID", "Sepsis_onset_next_6h"]]


def _load_events(
    file_path: Path,
    is_chart: bool,
    itemid_map: dict[str, list[int]],
    subject_ids: set[int],
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    all_itemids = sorted({int(item) for items in itemid_map.values() for item in items})
    reverse_map = {int(item): name for name, items in itemid_map.items() for item in items}

    if is_chart:
        usecols = ["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM", "VALUEUOM"]
    else:
        usecols = ["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM", "VALUEUOM"]

    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=chunksize):
        chunk = chunk[chunk["SUBJECT_ID"].isin(subject_ids)]
        chunk = chunk[chunk["ITEMID"].isin(all_itemids)]
        if chunk.empty:
            continue

        chunk["Timestamp"] = pd.to_datetime(chunk["CHARTTIME"], errors="coerce")
        chunk["VALUENUM"] = _safe_numeric(chunk["VALUENUM"])
        chunk = chunk.dropna(subset=["Timestamp", "VALUENUM"])
        if chunk.empty:
            continue

        chunk["variable"] = chunk["ITEMID"].map(reverse_map)
        chunk = chunk.dropna(subset=["variable"])
        parts.append(chunk[["SUBJECT_ID", "HADM_ID", "Timestamp", "variable", "VALUENUM", "VALUEUOM"]])

    if not parts:
        return pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "Timestamp", "variable", "VALUENUM", "VALUEUOM"])

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["SUBJECT_ID", "HADM_ID", "Timestamp"]).drop_duplicates(
        subset=["SUBJECT_ID", "HADM_ID", "Timestamp", "variable"],
        keep="last",
    )
    return out


def _pivot_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "Timestamp"])  # pragma: no cover

    wide = (
        events.pivot_table(
            index=["SUBJECT_ID", "HADM_ID", "Timestamp"],
            columns="variable",
            values="VALUENUM",
            aggfunc="last",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    return wide


def _normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Temp" in out.columns:
        temp_f_mask = out["Temp"] > 45.0
        out.loc[temp_f_mask, "Temp"] = (out.loc[temp_f_mask, "Temp"] - 32.0) * (5.0 / 9.0)

    if "Lactate" in out.columns:
        lactate_mgdl_mask = out["Lactate"] > 20.0
        out.loc[lactate_mgdl_mask, "Lactate"] = out.loc[lactate_mgdl_mask, "Lactate"] / 9.0

    return out


def _resample_per_admission(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    value_cols = ["HR", "SpO2", "RR", "Temp", "WBC", "CRP", "Platelets", "Lactate"]

    for (sid, hadm), group in df.groupby(["SUBJECT_ID", "HADM_ID"], sort=False):
        group = group.sort_values("Timestamp")
        if group.empty:
            continue

        idx = pd.date_range(group["Timestamp"].min(), group["Timestamp"].max(), freq=freq)
        if len(idx) < 2:
            continue

        local = group.set_index("Timestamp").reindex(idx)
        local.index.name = "Timestamp"
        local = local.reset_index()
        local["SUBJECT_ID"] = int(sid)
        local["HADM_ID"] = int(hadm)

        for col in ["HR", "SpO2", "RR", "Temp"]:
            if col in local.columns:
                local[col] = local[col].ffill().bfill()

        existing_cols = [col for col in value_cols if col in local.columns]
        for col in existing_cols:
            local[col] = _safe_numeric(local[col])
        pieces.append(local)

    if not pieces:
        return pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "Timestamp"] + value_cols)
    return pd.concat(pieces, ignore_index=True)


def generate_sparsity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["Patient_ID", "Timestamp"]).copy()
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
    out = out.dropna(subset=["Timestamp"]).copy()
    for lab in LABS:
        out[f"{lab}_measured_mask"] = out[lab].notna().astype(float)
        measured_ts = out["Timestamp"].where(out[lab].notna())
        last_measured = measured_ts.groupby(out["Patient_ID"], sort=False).ffill()
        hours_since = (out["Timestamp"] - last_measured).dt.total_seconds() / 3600.0
        from_start = (
            out["Timestamp"] - out.groupby("Patient_ID", sort=False)["Timestamp"].transform("min")
        ).dt.total_seconds() / 3600.0
        out[f"{lab}_hours_since_measured"] = hours_since.fillna(from_start).fillna(0.0).astype(float)
    return out


def align_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in MIMIC_BASE_COLS:
        if col not in out.columns:
            if col == "Hospital_ID":
                out[col] = "PIC"
            elif col == "Sepsis_onset_next_6h":
                out[col] = 0.0
            else:
                out[col] = np.nan

    sparsity_cols = [f"{lab}_{suffix}" for lab in LABS for suffix in ["measured_mask", "hours_since_measured"]]
    for col in sparsity_cols:
        if col not in out.columns:
            out[col] = 0.0

    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
    out = out.dropna(subset=["Timestamp"]).sort_values(["Patient_ID", "Timestamp"])
    return out[MIMIC_BASE_COLS + sparsity_cols]


def build_pic_aligned_dataset(base_dir: Path, max_subjects: int | None, output_path: Path) -> pd.DataFrame:
    cohort = build_pediatric_cohort(base_dir=base_dir, max_subjects=None)
    if cohort.empty:
        raise RuntimeError("PIC cohort is empty after pediatric/NICU filtering.")

    if max_subjects is not None and max_subjects > 0:
        candidate_ids = set(cohort["SUBJECT_ID"].astype(int).tolist())
        chart_itemids = sorted({item for key in ["HR", "SpO2", "RR", "Temp"] for item in PIC_ITEMIDS[key]})
        lab_itemids = sorted({item for key in ["WBC", "CRP", "Platelets", "Lactate"] for item in PIC_ITEMIDS[key]})
        available_ids = set(
            select_event_available_subjects(
                base_dir=base_dir,
                candidate_subject_ids=candidate_ids,
                chart_itemids=chart_itemids,
                lab_itemids=lab_itemids,
                max_subjects=len(candidate_ids),  # fetch all, then balance below
            )
        )
        # Balanced selection: up to half positives, half negatives (mirrors MIMIC build)
        sepsis_ids = set(cohort[cohort["Sepsis_onset_next_6h"] == 1.0]["SUBJECT_ID"].astype(int).tolist())
        avail_pos = sorted(available_ids & sepsis_ids)
        avail_neg = sorted(available_ids - sepsis_ids)
        n_pos = min(len(avail_pos), max_subjects // 2)
        n_neg = min(len(avail_neg), max_subjects - n_pos)
        keep_subjects = set(avail_pos[:n_pos] + avail_neg[:n_neg])
        cohort = cohort[cohort["SUBJECT_ID"].astype(int).isin(keep_subjects)].copy()

    subject_ids = set(cohort["SUBJECT_ID"].astype(int).tolist())
    chart_map = {k: PIC_ITEMIDS[k] for k in ["HR", "SpO2", "RR", "Temp"]}
    lab_map = {k: PIC_ITEMIDS[k] for k in ["WBC", "CRP", "Platelets", "Lactate"]}

    vitals_events = _load_events(base_dir / "CHARTEVENTS.csv", is_chart=True, itemid_map=chart_map, subject_ids=subject_ids)
    lab_events = _load_events(base_dir / "LABEVENTS.csv", is_chart=False, itemid_map=lab_map, subject_ids=subject_ids)

    vitals_wide = _pivot_events(vitals_events)
    labs_wide = _pivot_events(lab_events)

    # Inner-join on cohort to keep only PICU patients; drop the label column before
    # resampling so the uniform 5-min grid does not NaN-fill it.
    cohort_keys = cohort[["SUBJECT_ID", "HADM_ID"]].copy()
    merged = vitals_wide.merge(labs_wide, on=["SUBJECT_ID", "HADM_ID", "Timestamp"], how="outer")
    merged = merged.merge(cohort_keys, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    merged = _normalize_units(merged)
    merged = _resample_per_admission(merged, freq="5min")
    # Re-join labels AFTER resampling so every row gets the correct per-stay label
    merged = merged.merge(cohort, on=["SUBJECT_ID", "HADM_ID"], how="left")
    merged = merged.loc[:, ~merged.columns.duplicated()].copy()

    merged["Hospital_ID"] = "PIC"
    sid = pd.to_numeric(merged["SUBJECT_ID"], errors="coerce").fillna(-1).astype(int).astype(str)
    hadm = pd.to_numeric(merged["HADM_ID"], errors="coerce").fillna(-1).astype(int).astype(str)
    merged["Patient_ID"] = "PIC_" + sid + "_" + hadm
    merged["Birth_Weight"] = np.nan
    merged["Gestational_Age"] = np.nan

    # For PIC we only have a discharge-level sepsis diagnosis (no onset time).
    # Approximate MIMIC's 6-hour look-ahead window: mark the LAST 6 h of each
    # sepsis-positive stay as positive, and everything before that as negative.
    merged["Timestamp"] = pd.to_datetime(merged["Timestamp"], errors="coerce")
    if "Sepsis_onset_next_6h" in merged.columns:
        horizon_td = pd.Timedelta(hours=6)
        stay_end = merged.groupby(["SUBJECT_ID", "HADM_ID"])["Timestamp"].transform("max")
        merged["Sepsis_onset_next_6h"] = (
            (merged["Sepsis_onset_next_6h"] == 1.0)
            & (merged["Timestamp"] >= stay_end - horizon_td)
        ).astype(float)

    merged = generate_sparsity_features(merged)
    aligned = align_schema(merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_path, index=False)
    return aligned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess PIC data to mimic-compatible aligned CSV.")
    parser.add_argument(
        "--pic_dir",
        type=str,
        default="paediatric-intensive-care-database-1.1.0/V1.1.0",
        help="Path to PIC CSV directory.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/pic_nicu_aligned.csv",
        help="Output aligned CSV path.",
    )
    parser.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Optional cap for number of subjects (useful for quick smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pic_dir = Path(args.pic_dir)
    output_csv = Path(args.output_csv)

    print("Starting PIC schema alignment...")
    aligned = build_pic_aligned_dataset(base_dir=pic_dir, max_subjects=args.max_subjects, output_path=output_csv)
    print(f"Saved aligned PIC dataset: {output_csv}")
    print(f"Rows={len(aligned)} | Patients={aligned['Patient_ID'].nunique()} | PosRate={aligned['Sepsis_onset_next_6h'].mean():.6f}")
    print("Feature summary sanity check:")
    print(aligned[["HR", "SpO2", "RR", "Temp", "WBC", "CRP", "Platelets", "Lactate"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()
