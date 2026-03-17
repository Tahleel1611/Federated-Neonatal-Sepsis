from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    n_hospitals: int = 3
    n_patients_per_hospital: int = 20
    observation_hours: int = 24
    prediction_horizon_hours: int = 6
    hrv_window_minutes: int = 60
    time_step_minutes: int = 5
    # Real MIMIC-III data settings
    use_mimic: bool = False
    mimic_dir: str = "mimic-iii-clinical-database-1.4"
    mimic_max_stays: int | None = None  # None = use all neonatal stays
    mimic_prebuilt_csv: str | None = None  # optional cached extraction CSV


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
    batch_size: int = 32
    lr: float = 3e-4
    epochs: int = 3
    random_seed: int = 42
    loss_name: str = "focal"  # Options: "focal", "bce_weighted"
    focal_alpha: float = 0.80
    focal_gamma: float = 2.0
    pos_class_weight: float = 10.0
    max_grad_norm: float = 1.0
    target_sensitivity: float = 0.80
    default_eval_threshold: float = 0.50
    neg_to_pos_ratio: float = 3.0  # Phase 3: Downsampling ratio for training data only
    steps_per_epoch: int = 500
    log_every_n_batches: int = 50


@dataclass
class FederatedConfig:
    server_address: str = "127.0.0.1:8080"
    rounds: int = 3
    local_epochs: int = 1
    local_steps: int | None = None
    use_dp: bool = False
    dp_noise_multiplier: float = 1.0
    dp_max_grad_norm: float = 1.0
    delta: float = 1e-5
    pos_class_weight: float = 10.0
    target_sensitivity: float = 0.80
    default_eval_threshold: float = 0.50
    neg_to_pos_ratio: float = 3.0  # Phase 3: Downsampling ratio for training data only