from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_optional_str(name: str, default: str | None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_optional_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class DataConfig:
    n_hospitals: int = field(default_factory=lambda: _env_int("SEPSIS_N_HOSPITALS", 3))
    n_patients_per_hospital: int = field(default_factory=lambda: _env_int("SEPSIS_N_PATIENTS_PER_HOSPITAL", 20))
    observation_hours: int = field(default_factory=lambda: _env_int("SEPSIS_OBSERVATION_HOURS", 24))
    prediction_horizon_hours: int = field(default_factory=lambda: _env_int("SEPSIS_PREDICTION_HORIZON_HOURS", 6))
    hrv_window_minutes: int = field(default_factory=lambda: _env_int("SEPSIS_HRV_WINDOW_MINUTES", 60))
    time_step_minutes: int = field(default_factory=lambda: _env_int("SEPSIS_TIME_STEP_MINUTES", 5))
    use_mimic: bool = field(default_factory=lambda: _env_bool("SEPSIS_USE_MIMIC", False))
    mimic_dir: str = field(default_factory=lambda: _env_str("SEPSIS_MIMIC_DIR", "mimic-iii-clinical-database-1.4"))
    mimic_max_stays: int | None = field(default_factory=lambda: _env_optional_int("SEPSIS_MIMIC_MAX_STAYS", None))
    mimic_prebuilt_csv: str | None = field(default_factory=lambda: _env_optional_str("SEPSIS_MIMIC_PREBUILT_CSV", None))


@dataclass
class ModelConfig:
    input_size: int = 10
    d_model: int = 64
    num_heads: int = 4
    transformer_layers: int = 2
    lstm_hidden: int = 64
    lstm_layers: int = 1
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = field(default_factory=lambda: _env_int("SEPSIS_BATCH_SIZE", 32))
    lr: float = field(default_factory=lambda: _env_float("SEPSIS_LR", 3e-4))
    epochs: int = field(default_factory=lambda: _env_int("SEPSIS_EPOCHS", 3))
    random_seed: int = field(default_factory=lambda: _env_int("SEPSIS_RANDOM_SEED", 42))
    loss_name: str = field(default_factory=lambda: _env_str("SEPSIS_LOSS_NAME", "focal"))
    focal_alpha: float = field(default_factory=lambda: _env_float("SEPSIS_FOCAL_ALPHA", 0.80))
    focal_gamma: float = field(default_factory=lambda: _env_float("SEPSIS_FOCAL_GAMMA", 2.0))
    pos_class_weight: float = field(default_factory=lambda: _env_float("SEPSIS_POS_CLASS_WEIGHT", 10.0))
    max_grad_norm: float = field(default_factory=lambda: _env_float("SEPSIS_MAX_GRAD_NORM", 1.0))
    target_sensitivity: float = field(default_factory=lambda: _env_float("SEPSIS_TARGET_SENSITIVITY", 0.80))
    default_eval_threshold: float = field(default_factory=lambda: _env_float("SEPSIS_DEFAULT_EVAL_THRESHOLD", 0.50))
    neg_to_pos_ratio: float = field(default_factory=lambda: _env_float("SEPSIS_NEG_TO_POS_RATIO", 3.0))
    steps_per_epoch: int = field(default_factory=lambda: _env_int("SEPSIS_STEPS_PER_EPOCH", 500))
    log_every_n_batches: int = field(default_factory=lambda: _env_int("SEPSIS_LOG_EVERY_N_BATCHES", 50))


@dataclass
class FederatedConfig:
    server_address: str = field(default_factory=lambda: _env_str("SEPSIS_SERVER_ADDRESS", "127.0.0.1:8080"))
    rounds: int = field(default_factory=lambda: _env_int("SEPSIS_ROUNDS", 3))
    local_epochs: int = field(default_factory=lambda: _env_int("SEPSIS_LOCAL_EPOCHS", 1))
    local_steps: int | None = field(default_factory=lambda: _env_optional_int("SEPSIS_LOCAL_STEPS", None))
    use_dp: bool = field(default_factory=lambda: _env_bool("SEPSIS_USE_DP", False))
    dp_noise_multiplier: float = field(default_factory=lambda: _env_float("SEPSIS_DP_NOISE_MULTIPLIER", 1.0))
    dp_max_grad_norm: float = field(default_factory=lambda: _env_float("SEPSIS_DP_MAX_GRAD_NORM", 1.0))
    delta: float = field(default_factory=lambda: _env_float("SEPSIS_DP_DELTA", 1e-5))
    pos_class_weight: float = field(default_factory=lambda: _env_float("SEPSIS_FED_POS_CLASS_WEIGHT", 10.0))
    target_sensitivity: float = field(default_factory=lambda: _env_float("SEPSIS_FED_TARGET_SENSITIVITY", 0.80))
    default_eval_threshold: float = field(default_factory=lambda: _env_float("SEPSIS_FED_DEFAULT_EVAL_THRESHOLD", 0.50))
    neg_to_pos_ratio: float = field(default_factory=lambda: _env_float("SEPSIS_FED_NEG_TO_POS_RATIO", 3.0))