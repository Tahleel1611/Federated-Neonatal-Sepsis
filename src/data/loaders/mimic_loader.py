"""
MIMIC-III Neonatal NICU data extractor.

Reads raw MIMIC-III .csv.gz tables and produces a DataFrame with the
same schema as ``generate_synthetic_nicu_logs``:

    Hospital_ID | Patient_ID | Timestamp | HR | SpO2 | RR | Temp |
    WBC | CRP | Platelets | Lactate | Birth_Weight | Gestational_Age |
    Sepsis_onset_next_6h

Usage
-----
    from src.data.loaders.mimic_loader import build_mimic_nicu_dataset
    df = build_mimic_nicu_dataset("path/to/mimic-iii-clinical-database-1.4")

Or via CLI:
    python -m src.data.loaders.mimic_loader \
        --mimic_dir mimic-iii-clinical-database-1.4 \
        --out_csv results/mimic_nicu_dataset.csv
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.constants import (
    KEY_COLUMNS,
    LAB_COLUMNS,
    STATIC_COLUMNS,
    TARGET_COLUMN,
    VITAL_COLUMNS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

# ---------------------------------------------------------------------------
# CHARTEVENTS item IDs — neonatal-specific (mined from D_ITEMS + live data)
# ---------------------------------------------------------------------------
# Heart Rate
HR_ITEMS = {211, 220045, 3494}          # 3494=Lowest HR (backup)
# O2 Saturation — neonatal monitors report SaO2 (834) not generic SpO2
SPO2_ITEMS = {834, 646, 220277, 3495,   # 834=SaO2, 3495=Lowest SaO2
              8208, 8209}               # 8208/8209=Post/Preductal O2 Sat
# Respiratory Rate — neonatal NICU uses item 3603, plus CareVue/MV fallbacks
RR_ITEMS = {3603, 615, 618, 614, 619,
            220210, 224689, 224690}
# Temperature °C (neonatal: skin probe 3655, standard 676/677/223762)
TEMP_C_ITEMS = {3655, 676, 677, 223762}
# Temperature °F (neonatal: axillary 3652, rectal 3654, plus 678/679/223761)
TEMP_F_ITEMS = {3652, 3654, 678, 679, 223761}
# Birth weight — neonatal chart uses kg (3723); convert ×1000 → grams below
BWT_KG_ITEMS  = {3723, 3580}            # 3580=Present Weight (kg)
BWT_GRAM_ITEMS = {4183}                 # rare grams item
BWT_LBS_ITEMS  = {829}                  # lbs → grams
# Gestational age (weeks)
GEST_AGE_ITEMS = {1394, 1340, 1337}

CHART_VITAL_ITEMS: set[int] = (
    HR_ITEMS | SPO2_ITEMS | RR_ITEMS | TEMP_C_ITEMS | TEMP_F_ITEMS
    | BWT_KG_ITEMS | BWT_GRAM_ITEMS | BWT_LBS_ITEMS | GEST_AGE_ITEMS
)

# ---------------------------------------------------------------------------
# LABEVENTS item IDs
# ---------------------------------------------------------------------------
LAB_ITEM_MAP: dict[int, str] = {
    51301: "WBC",       # White Blood Cells ×10³/µL (Hematology)
    51300: "WBC",       # WBC — alternate itemid
    50889: "CRP",       # C-Reactive Protein mg/L (sparse in neonates → imputed)
    50963: "CRP",       # CRP alternate
    51265: "Platelets", # Platelet Count ×10³/µL
    51244: "WBC",       # Lymphocytes — used as WBC surrogate when WBC absent
    50813: "Lactate",   # Lactate mmol/L (sparse in neonates → imputed)
    50756: "Lactate",   # Lactate alternate
}

LAB_ITEMIDS: set[int] = set(LAB_ITEM_MAP.keys())

# ---------------------------------------------------------------------------
# Neonatal sepsis ICD-9 codes
# ---------------------------------------------------------------------------
SEPSIS_ICD9_PREFIXES = (
    "77181",  # Neonatal group B streptococcal septicemia
    "77183",  # Bacteremia of newborn
    "77189",  # Other infections specific to perinatal period
    "7719",   # Unspecified infection of newborn
    "03891",  # Gram-negative septicemia NEC
    "03840",  # Septicemia NEC
    "0389",   # Unspecified septicemia
    "99591",  # Sepsis
    "99592",  # Severe sepsis
)

# Normal physiological ranges for neonates (used to clip outliers)
VITAL_CLIP: dict[str, tuple[float, float]] = {
    "HR":        (60,  260),
    "SpO2":      (50,  100),
    "RR":        (10,  120),
    "Temp":      (34.0, 41.0),
}
LAB_CLIP: dict[str, tuple[float, float]] = {
    "WBC":       (0.1,  80.0),
    "CRP":       (0.0,  300.0),
    "Platelets": (5.0,  1500.0),
    "Lactate":   (0.1,  25.0),
}

CHUNK_SIZE = 500_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gz(mimic_dir: str | Path, table: str) -> Path:
    """Return the path to a MIMIC-III .csv.gz file."""
    p = Path(mimic_dir) / f"{table.upper()}.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"MIMIC-III table not found: {p}")
    return p


def _read_full(mimic_dir: str | Path, table: str, usecols: list[str] | None = None) -> pd.DataFrame:
    logger.info("Reading %s …", table.upper())
    return pd.read_csv(_gz(mimic_dir, table), usecols=usecols, low_memory=False)


def _read_chunks(
    path: Path,
    usecols: list[str],
    filter_col: str,
    filter_values: Iterable,
) -> pd.DataFrame:
    """Read a large CSV.gz in chunks, keeping only rows where
    ``filter_col`` is in ``filter_values``."""
    fv = set(filter_values)
    chunks = []
    reader = pd.read_csv(
        path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False
    )
    for chunk in reader:
        mask = chunk[filter_col].isin(fv)
        if mask.any():
            chunks.append(chunk.loc[mask])
    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True)


def _ensure_neonate_cache(
    mimic_dir: Path,
    icustay_ids: Iterable[int],
    hadm_ids: Iterable[int],
    cache_dir: Path,
) -> tuple[Path, Path]:
    """
    Pre-filter the large CHARTEVENTS and LABEVENTS tables to neonatal rows
    and persist compressed CSVs in ``cache_dir``.  Subsequent calls that
    find the cache files skip the expensive full-scan entirely.

    Returns (chart_cache_path, lab_cache_path).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    chart_cache = cache_dir / "chartevents_neo.csv.gz"
    lab_cache   = cache_dir / "labevents_neo.csv.gz"

    if not chart_cache.exists():
        logger.info(
            "Building CHARTEVENTS neonatal cache (one-time, ~8 min) → %s …", chart_cache
        )
        raw = _read_chunks(
            _gz(mimic_dir, "CHARTEVENTS"),
            ["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM", "ERROR"],
            "ICUSTAY_ID",
            icustay_ids,
        )
        raw.to_csv(chart_cache, index=False, compression="gzip")
        logger.info("CHARTEVENTS cache written: %d rows", len(raw))
    else:
        logger.info("Using cached CHARTEVENTS: %s", chart_cache)

    if not lab_cache.exists():
        logger.info("Building LABEVENTS neonatal cache → %s …", lab_cache)
        raw = _read_chunks(
            _gz(mimic_dir, "LABEVENTS"),
            ["HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
            "HADM_ID",
            hadm_ids,
        )
        raw.to_csv(lab_cache, index=False, compression="gzip")
        logger.info("LABEVENTS cache written: %d rows", len(raw))
    else:
        logger.info("Using cached LABEVENTS: %s", lab_cache)

    return chart_cache, lab_cache


# ---------------------------------------------------------------------------
# Step 1: Identify neonatal ICU stays
# ---------------------------------------------------------------------------
def extract_neonatal_stays(mimic_dir: str | Path) -> pd.DataFrame:
    """
    Return a DataFrame of neonatal ICU stays with columns:
        SUBJECT_ID, HADM_ID, ICUSTAY_ID, CARE_UNIT,
        INTIME, OUTTIME, Hospital_ID, Patient_ID,
        sepsis_flag
    """
    # --- Patients ---
    patients = _read_full(
        mimic_dir, "PATIENTS",
        usecols=["SUBJECT_ID", "DOB", "GENDER"],
    )
    patients["DOB"] = pd.to_datetime(patients["DOB"])

    # --- Admissions: keep NEWBORN type ---
    admissions = _read_full(
        mimic_dir, "ADMISSIONS",
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "ADMISSION_TYPE"],
    )
    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"])
    newborn_hadm = admissions[
        admissions["ADMISSION_TYPE"].str.upper() == "NEWBORN"
    ][["SUBJECT_ID", "HADM_ID", "ADMITTIME"]].copy()

    logger.info("Newborn admissions: %d", len(newborn_hadm))

    if newborn_hadm.empty:
        # Fallback: identify by age at admission (< 30 days)
        adm_merged = admissions.merge(patients[["SUBJECT_ID", "DOB"]], on="SUBJECT_ID")
        adm_merged["age_days"] = (
            adm_merged["ADMITTIME"] - adm_merged["DOB"]
        ).dt.total_seconds() / 86400
        newborn_hadm = adm_merged[adm_merged["age_days"] < 30][
            ["SUBJECT_ID", "HADM_ID", "ADMITTIME"]
        ].copy()
        logger.info("Newborn (age<30d) admissions: %d", len(newborn_hadm))

    # --- ICU Stays ---
    icustays = _read_full(
        mimic_dir, "ICUSTAYS",
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "FIRST_CAREUNIT", "INTIME", "OUTTIME", "LOS"],
    )
    icustays["INTIME"] = pd.to_datetime(icustays["INTIME"])
    icustays["OUTTIME"] = pd.to_datetime(icustays["OUTTIME"])

    neo_icu = icustays[icustays["HADM_ID"].isin(newborn_hadm["HADM_ID"])].copy()
    logger.info("Neonatal ICU stays: %d", len(neo_icu))

    # Assign hospital partition by SUBJECT_ID hash (balanced 3-way split)
    neo_icu["Hospital_ID"] = (neo_icu["SUBJECT_ID"] % 3).map(
        {0: "H1", 1: "H2", 2: "H3"}
    )
    neo_icu["Patient_ID"] = "SUBJ_" + neo_icu["SUBJECT_ID"].astype(str)

    # --- Sepsis labels via DIAGNOSES_ICD ---
    logger.info("Loading DIAGNOSES_ICD for sepsis labels …")
    diag = _read_full(
        mimic_dir, "DIAGNOSES_ICD",
        usecols=["HADM_ID", "ICD9_CODE"],
    )
    diag["ICD9_CODE"] = diag["ICD9_CODE"].astype(str).str.strip().str.replace(".", "", regex=False)
    diag["sepsis_flag"] = diag["ICD9_CODE"].str.startswith(SEPSIS_ICD9_PREFIXES)
    sepsis_hadm = set(diag.loc[diag["sepsis_flag"], "HADM_ID"].unique())
    neo_icu["sepsis_flag"] = neo_icu["HADM_ID"].isin(sepsis_hadm)
    logger.info(
        "Sepsis-positive stays: %d / %d",
        neo_icu["sepsis_flag"].sum(), len(neo_icu),
    )

    return neo_icu.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: Extract vital signs from CHARTEVENTS
# ---------------------------------------------------------------------------
def extract_vitals(chart_path: Path, icustay_ids: Iterable[int]) -> pd.DataFrame:
    """
    Return vital-sign time-series from a (pre-filtered) CHARTEVENTS file.
    Columns: ICUSTAY_ID, CHARTTIME, HR, SpO2, RR, Temp,
             Birth_Weight_raw, Gestational_Age_raw
    """
    logger.info("Loading CHARTEVENTS from cache/source …")
    path = chart_path
    usecols = ["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM", "ERROR"]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    selected_icustay_ids = set(int(x) for x in icustay_ids)
    raw = raw[raw["ICUSTAY_ID"].isin(selected_icustay_ids)].copy()
    logger.info("CHARTEVENTS rows for selected stays: %d", len(raw))

    # Drop error rows
    raw = raw[raw["ERROR"].isna() | (raw["ERROR"] == 0)].copy()
    raw = raw[raw["ITEMID"].isin(CHART_VITAL_ITEMS)].copy()
    raw["CHARTTIME"] = pd.to_datetime(raw["CHARTTIME"])
    raw = raw.dropna(subset=["VALUENUM", "CHARTTIME", "ICUSTAY_ID"])

    # --- Map itemid → canonical feature name ---
    def _map_feature(row: pd.Series) -> str:
        iid = int(row["ITEMID"])
        if iid in HR_ITEMS:
            return "HR"
        if iid in SPO2_ITEMS:
            return "SpO2"
        if iid in RR_ITEMS:
            return "RR"
        if iid in TEMP_C_ITEMS:
            return "Temp_C"
        if iid in TEMP_F_ITEMS:
            return "Temp_F"
        if iid in BWT_KG_ITEMS:
            return "BWT_kg"
        if iid in BWT_GRAM_ITEMS:
            return "BWT_g"
        if iid in BWT_LBS_ITEMS:
            return "BWT_lbs"
        if iid in GEST_AGE_ITEMS:
            return "GEST"
        return "UNKNOWN"

    raw["feature"] = raw.apply(_map_feature, axis=1)

    # Convert Fahrenheit → Celsius
    temp_f = raw["feature"] == "Temp_F"
    raw.loc[temp_f, "VALUENUM"] = (raw.loc[temp_f, "VALUENUM"] - 32) * 5 / 9
    raw.loc[temp_f, "feature"] = "Temp_C"

    # Convert kg → grams
    bwt_kg = raw["feature"] == "BWT_kg"
    raw.loc[bwt_kg, "VALUENUM"] = raw.loc[bwt_kg, "VALUENUM"] * 1000
    raw.loc[bwt_kg, "feature"] = "BWT_g"

    # Convert lbs → grams
    bwt_lbs = raw["feature"] == "BWT_lbs"
    raw.loc[bwt_lbs, "VALUENUM"] = raw.loc[bwt_lbs, "VALUENUM"] * 453.592
    raw.loc[bwt_lbs, "feature"] = "BWT_g"

    # Rename accumulated features
    rename = {"Temp_C": "Temp", "BWT_g": "Birth_Weight_raw", "GEST": "Gestational_Age_raw"}
    raw["feature"] = raw["feature"].replace(rename)

    # Pivot to wide format: one row per (ICUSTAY_ID, CHARTTIME)
    vitals_wide = (
        raw.groupby(["ICUSTAY_ID", "CHARTTIME", "feature"])["VALUENUM"]
        .mean()  # if itemid duplicated for same timestamp
        .unstack("feature")
        .reset_index()
    )
    # Ensure columns exist
    for col in ["HR", "SpO2", "RR", "Temp", "Birth_Weight_raw", "Gestational_Age_raw"]:
        if col not in vitals_wide.columns:
            vitals_wide[col] = np.nan

    return vitals_wide


# ---------------------------------------------------------------------------
# Step 3: Extract lab values from LABEVENTS
# ---------------------------------------------------------------------------
def extract_labs(lab_path: Path, hadm_ids: Iterable[int]) -> pd.DataFrame:
    """
    Return lab-value time-series from a (pre-filtered) LABEVENTS file.
    Columns: HADM_ID, CHARTTIME, WBC, CRP, Platelets, Lactate
    """
    logger.info("Loading LABEVENTS from cache/source …")
    usecols = ["HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"]
    raw = pd.read_csv(lab_path, usecols=usecols, low_memory=False)
    selected_hadm_ids = set(int(x) for x in hadm_ids)
    raw = raw[raw["HADM_ID"].isin(selected_hadm_ids)].copy()
    logger.info("LABEVENTS rows for selected stays: %d", len(raw))

    raw = raw[raw["ITEMID"].isin(LAB_ITEMIDS)].copy()
    raw["CHARTTIME"] = pd.to_datetime(raw["CHARTTIME"])
    raw = raw.dropna(subset=["VALUENUM", "CHARTTIME", "HADM_ID"])
    raw["feature"] = raw["ITEMID"].map(LAB_ITEM_MAP)

    labs_wide = (
        raw.groupby(["HADM_ID", "CHARTTIME", "feature"])["VALUENUM"]
        .mean()
        .unstack("feature")
        .reset_index()
    )
    for col in ["WBC", "CRP", "Platelets", "Lactate"]:
        if col not in labs_wide.columns:
            labs_wide[col] = np.nan

    return labs_wide


# ---------------------------------------------------------------------------
# Step 4: Assemble time-series + labels
# ---------------------------------------------------------------------------
def _build_stay_timeseries(
    stay: pd.Series,
    vitals: pd.DataFrame,
    labs: pd.DataFrame,
    prediction_horizon_h: int = 6,
    resample_freq: str = "5min",
) -> pd.DataFrame:
    """
    For one ICU stay, merge vitals + labs onto a regular time grid,
    forward-fill, then create the Sepsis_onset_next_6h label.
    """
    icustay_id = stay["ICUSTAY_ID"]
    hadm_id = stay["HADM_ID"]
    intime: pd.Timestamp = stay["INTIME"]
    outtime: pd.Timestamp = stay["OUTTIME"]
    sepsis_flag: bool = stay["sepsis_flag"]

    if pd.isna(intime) or pd.isna(outtime) or outtime <= intime:
        return pd.DataFrame()

    freq_delta = pd.to_timedelta(resample_freq)

    # Build regular grid (aligned to frequency boundaries)
    grid_start = intime.floor(resample_freq)
    grid_end = outtime.ceil(resample_freq)
    grid = pd.date_range(grid_start, grid_end, freq=resample_freq, name="Timestamp")
    if len(grid) < 2:
        return pd.DataFrame()

    df_grid = pd.DataFrame(index=grid).reset_index()

    # --- Vitals ---
    sv = vitals[vitals["ICUSTAY_ID"] == icustay_id].copy()
    sv = sv.rename(columns={"CHARTTIME": "Timestamp"})
    sv = sv.drop(columns=["ICUSTAY_ID"], errors="ignore")
    if not sv.empty:
        sv["Timestamp"] = pd.to_datetime(sv["Timestamp"]).dt.floor(resample_freq)
        sv = sv.groupby("Timestamp", as_index=False)[
            ["HR", "SpO2", "RR", "Temp", "Birth_Weight_raw", "Gestational_Age_raw"]
        ].mean()
        df_grid = df_grid.merge(sv, on="Timestamp", how="left")
    else:
        for col in ["HR", "SpO2", "RR", "Temp", "Birth_Weight_raw", "Gestational_Age_raw"]:
            df_grid[col] = np.nan

    # --- Labs ---
    sl = labs[labs["HADM_ID"] == hadm_id].copy()
    sl = sl.rename(columns={"CHARTTIME": "Timestamp"})
    sl = sl.drop(columns=["HADM_ID"], errors="ignore")
    if not sl.empty:
        sl["Timestamp"] = pd.to_datetime(sl["Timestamp"]).dt.floor(resample_freq)
        sl = sl.groupby("Timestamp", as_index=False)[["WBC", "CRP", "Platelets", "Lactate"]].mean()
        df_grid = df_grid.merge(sl, on="Timestamp", how="left")
    else:
        for col in ["WBC", "CRP", "Platelets", "Lactate"]:
            df_grid[col] = np.nan

    # Forward-fill then backward-fill within patient
    fill_cols = ["HR", "SpO2", "RR", "Temp", "WBC", "CRP", "Platelets", "Lactate"]
    for col in fill_cols:
        if col in df_grid.columns:
            df_grid[col] = df_grid[col].ffill().bfill()

    # --- Static features: take first non-null value across entire stay ---
    if "Birth_Weight_raw" in df_grid.columns:
        bw = df_grid["Birth_Weight_raw"].dropna()
        bw_val = float(bw.median()) if not bw.empty else 2500.0
    else:
        bw_val = 2500.0

    if "Gestational_Age_raw" in df_grid.columns:
        ga = df_grid["Gestational_Age_raw"].dropna()
        ga_val = float(ga.median()) if not ga.empty else 36.0
    else:
        ga_val = 36.0

    # Birth_Weight_raw stores grams; convert to kg consistent with synthetic
    df_grid["Birth_Weight"] = np.clip(bw_val / 1000.0, 0.3, 5.5)
    df_grid["Gestational_Age"] = np.clip(ga_val, 22.0, 44.0)

    # --- Label: sepsis onset within next prediction_horizon_h ---
    # Strategy: if the admission is labelled septic, all time steps in the
    # final prediction_horizon_h window are positive; earlier steps negative.
    # This mirrors the synthetic generator's labelling approach.
    horizon_minutes = prediction_horizon_h * 60
    step_minutes = max(1, int(freq_delta.total_seconds() // 60))
    horizon_steps = max(1, int(horizon_minutes / step_minutes))
    n = len(df_grid)
    label = np.zeros(n, dtype=int)
    if sepsis_flag:
        onset_step = max(0, n - horizon_steps)
        label[onset_step:] = 1
    df_grid[TARGET_COLUMN] = label

    # --- Keys ---
    df_grid["Hospital_ID"] = stay["Hospital_ID"]
    df_grid["Patient_ID"] = stay["Patient_ID"]

    # --- Clip outliers ---
    for col, (lo, hi) in VITAL_CLIP.items():
        if col in df_grid.columns:
            df_grid[col] = df_grid[col].clip(lo, hi)
    for col, (lo, hi) in LAB_CLIP.items():
        if col in df_grid.columns:
            df_grid[col] = df_grid[col].clip(lo, hi)

    # Ensure required columns exist (fill remaining NaN with median or 0)
    required_numeric = VITAL_COLUMNS + LAB_COLUMNS + STATIC_COLUMNS
    for col in required_numeric:
        if col not in df_grid.columns:
            df_grid[col] = np.nan

    return df_grid


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------
def build_mimic_nicu_dataset(
    mimic_dir: str | Path,
    prediction_horizon_h: int = 6,
    resample_freq: str = "5min",
    max_stays: int | None = None,
) -> pd.DataFrame:
    """
    Full pipeline: MIMIC-III tables → neonatal NICU DataFrame.

    Parameters
    ----------
    mimic_dir : path to the MIMIC-III directory containing .csv.gz files
    prediction_horizon_h : hours ahead for sepsis label window
    resample_freq : pandas offset alias for time-grid spacing
    max_stays : if set, only process this many stays (for quick debugging)

    Returns
    -------
    DataFrame with columns matching the synthetic loader schema.
    """
    mimic_dir = Path(mimic_dir)
    cache_dir = mimic_dir.parent / "results" / "cache"

    # Step 1: all neonatal stays (needed to build the full-coverage cache)
    stays_all = extract_neonatal_stays(mimic_dir)
    stays_all = stays_all.dropna(subset=["ICUSTAY_ID", "HADM_ID", "INTIME", "OUTTIME"]).copy()
    stays_all = stays_all[stays_all["OUTTIME"] > stays_all["INTIME"]].copy()

    all_icustay_ids = stays_all["ICUSTAY_ID"].astype(int).tolist()
    all_hadm_ids    = stays_all["HADM_ID"].astype(int).tolist()

    # Build/load neonatal-filtered cache (one-time, then instant)
    chart_cache, lab_cache = _ensure_neonate_cache(
        mimic_dir, all_icustay_ids, all_hadm_ids, cache_dir
    )

    # Apply max_stays limit AFTER cache is built
    stays = stays_all.copy()
    if max_stays is not None:
        # Balanced selection: include sepsis-positive stays first, then negatives.
        pos_stays = stays[stays["sepsis_flag"]].copy()
        neg_stays = stays[~stays["sepsis_flag"]].copy()
        n_pos = min(len(pos_stays), max_stays // 2)
        n_neg = max_stays - n_pos
        stays = pd.concat([pos_stays.head(n_pos), neg_stays.head(n_neg)], ignore_index=True)
        if len(stays) < max_stays:
            remaining = max_stays - len(stays)
            extra_pos = pos_stays.iloc[n_pos : n_pos + remaining]
            stays = pd.concat([stays, extra_pos], ignore_index=True)
        logger.info("Limiting to %d stays (--max_stays).", max_stays)
        logger.info(
            "Selected stays breakdown: total=%d | sepsis-positive=%d | sepsis-negative=%d",
            len(stays),
            int(stays["sepsis_flag"].sum()),
            int((~stays["sepsis_flag"]).sum()),
        )

    icustay_ids = stays["ICUSTAY_ID"].astype(int).tolist()
    hadm_ids    = stays["HADM_ID"].astype(int).tolist()

    # Step 2: vitals from cache
    vitals = extract_vitals(chart_cache, icustay_ids)

    # Step 3: labs from cache
    labs = extract_labs(lab_cache, hadm_ids)

    # Step 4: assemble per-stay time-series
    logger.info("Assembling %d stay time-series …", len(stays))
    all_dfs: list[pd.DataFrame] = []
    for _, stay in stays.iterrows():
        ts_df = _build_stay_timeseries(
            stay, vitals, labs,
            prediction_horizon_h=prediction_horizon_h,
            resample_freq=resample_freq,
        )
        if len(ts_df) >= 12:
            all_dfs.append(ts_df)

    if not all_dfs:
        raise RuntimeError(
            "No valid stay time-series assembled. "
            "Check that MIMIC-III tables contain neonatal admissions."
        )

    df = pd.concat(all_dfs, ignore_index=True)

    # Final schema ordering
    out_cols = KEY_COLUMNS + VITAL_COLUMNS + LAB_COLUMNS + STATIC_COLUMNS + [TARGET_COLUMN]
    for col in out_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[out_cols]

    # Remaining NaN → column median (safety net)
    num_cols = VITAL_COLUMNS + LAB_COLUMNS + STATIC_COLUMNS
    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

    # If any feature is still all-NaN (median is NaN), apply physiologic defaults.
    fallback_defaults = {
        "HR": 140.0,
        "SpO2": 96.0,
        "RR": 45.0,
        "Temp": 36.7,
        "WBC": 12.0,
        "CRP": 1.0,
        "Platelets": 250.0,
        "Lactate": 1.5,
        "Birth_Weight": 2.5,
        "Gestational_Age": 36.0,
    }
    for col, default in fallback_defaults.items():
        if col in df.columns and df[col].isna().all():
            df[col] = default

    logger.info(
        "Dataset ready: %d rows | %d patients | %.1f%% sepsis-positive",
        len(df),
        df["Patient_ID"].nunique(),
        100 * df[TARGET_COLUMN].mean(),
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MIMIC-III neonatal NICU dataset.")
    parser.add_argument(
        "--mimic_dir",
        default="mimic-iii-clinical-database-1.4",
        help="Path to MIMIC-III directory with .csv.gz tables.",
    )
    parser.add_argument(
        "--out_csv",
        default="results/mimic_nicu_dataset.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--max_stays",
        type=int,
        default=None,
        help="Limit number of ICU stays (for quick testing).",
    )
    parser.add_argument(
        "--prediction_horizon_h",
        type=int,
        default=6,
        help="Prediction horizon in hours for sepsis label.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = build_mimic_nicu_dataset(
        mimic_dir=args.mimic_dir,
        prediction_horizon_h=args.prediction_horizon_h,
        max_stays=args.max_stays,
    )
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved → %s  (%d rows)", out, len(df))


if __name__ == "__main__":
    main()
