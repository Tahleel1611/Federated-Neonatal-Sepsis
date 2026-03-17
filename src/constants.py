VITAL_COLUMNS = ["HR", "SpO2", "RR", "Temp"]
LAB_COLUMNS = ["WBC", "CRP", "Platelets", "Lactate"]
LAB_MASK_COLUMNS = [f"{col}_measured_mask" for col in LAB_COLUMNS]
LAB_AGE_COLUMNS = [f"{col}_hours_since_measured" for col in LAB_COLUMNS]
STATIC_COLUMNS = ["Birth_Weight", "Gestational_Age"]
KEY_COLUMNS = ["Hospital_ID", "Patient_ID", "Timestamp"]
HRV_COLUMNS = ["HRV_SDNN", "HRV_RMSSD"]
TARGET_COLUMN = "Sepsis_onset_next_6h"
MODEL_FEATURE_COLUMNS = VITAL_COLUMNS + LAB_COLUMNS + LAB_MASK_COLUMNS + LAB_AGE_COLUMNS + HRV_COLUMNS + STATIC_COLUMNS


MIMIC_COMPATIBLE_MAP = {
    "heart_rate": "HR",
    "spo2": "SpO2",
    "resp_rate": "RR",
    "temperature": "Temp",
    "birth_weight": "Birth_Weight",
    "gest_age_weeks": "Gestational_Age",
}