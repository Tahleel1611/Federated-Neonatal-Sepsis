from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import ModelConfig
from src.constants import MODEL_FEATURE_COLUMNS
from src.models.transformer_lstm import TransformerLSTMSepsisModel

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_CANDIDATES = (
    "results/federated_global_model_final.pt",
    "results/centralized_model_latest.pt",
)

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")


class DeploymentError(RuntimeError):
    pass


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


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    checkpoint_path: Path
    seq_len_steps: int
    threshold: float
    is_dummy: bool = False


def resolve_checkpoint_path(explicit: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_path = _env_optional_str("SEPSIS_CHECKPOINT_PATH", None)
    if env_path:
        candidates.append(Path(env_path))
    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        candidates.append(Path(candidate))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _make_dummy_bundle() -> ModelBundle:
    """Return a randomly-initialised model bundle for demo / health-check purposes."""
    logger.warning(
        "No model checkpoint found. Starting in DEMO mode with a random model. "
        "Predictions will NOT be clinically meaningful."
    )
    model_cfg = ModelConfig()
    model = TransformerLSTMSepsisModel(
        input_size=len(MODEL_FEATURE_COLUMNS),
        d_model=model_cfg.d_model,
        num_heads=model_cfg.num_heads,
        transformer_layers=model_cfg.transformer_layers,
        lstm_hidden=model_cfg.lstm_hidden,
        lstm_layers=model_cfg.lstm_layers,
        dropout=0.0,
    )
    model.eval()
    return ModelBundle(
        model=model,
        checkpoint_path=Path("demo_dummy"),
        seq_len_steps=_env_int("SEPSIS_SEQ_LEN_STEPS", 12),
        threshold=_env_float("SEPSIS_PREDICTION_THRESHOLD", 0.5),
        is_dummy=True,
    )


def load_model_bundle(checkpoint_path: str | None = None) -> ModelBundle:
    resolved_path = resolve_checkpoint_path(checkpoint_path)

    if resolved_path is None:
        return _make_dummy_bundle()

    try:
        checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(resolved_path, map_location="cpu")

    model_cfg = ModelConfig()
    model = TransformerLSTMSepsisModel(
        input_size=int(checkpoint.get("input_size", len(MODEL_FEATURE_COLUMNS))),
        d_model=int(checkpoint.get("d_model", model_cfg.d_model)),
        num_heads=int(checkpoint.get("num_heads", model_cfg.num_heads)),
        transformer_layers=int(checkpoint.get("transformer_layers", model_cfg.transformer_layers)),
        lstm_hidden=int(checkpoint.get("lstm_hidden", model_cfg.lstm_hidden)),
        lstm_layers=int(checkpoint.get("lstm_layers", model_cfg.lstm_layers)),
        dropout=float(checkpoint.get("dropout", model_cfg.dropout)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    seq_len_steps = int(
        checkpoint.get(
            "seq_len_steps",
            _env_int("SEPSIS_SEQ_LEN_STEPS", 12),
        )
    )
    threshold = float(
        checkpoint.get(
            "best_threshold",
            _env_float("SEPSIS_PREDICTION_THRESHOLD", 0.5),
        )
    )
    return ModelBundle(
        model=model,
        checkpoint_path=resolved_path,
        seq_len_steps=seq_len_steps,
        threshold=threshold,
        is_dummy=False,
    )


def rows_to_sequence(rows: list[dict[str, Any]], seq_len_steps: int) -> np.ndarray:
    if not rows:
        raise DeploymentError("At least one row is required for prediction.")
    frame = pd.DataFrame(rows)
    if "Timestamp" in frame.columns:
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
        frame = frame.sort_values("Timestamp", kind="stable")
    missing = [column for column in MODEL_FEATURE_COLUMNS if column not in frame.columns]
    for column in missing:
        frame[column] = 0.0
    feature_frame = frame[MODEL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if len(feature_frame) < seq_len_steps:
        pad = pd.DataFrame(
            np.zeros((seq_len_steps - len(feature_frame), len(MODEL_FEATURE_COLUMNS))),
            columns=MODEL_FEATURE_COLUMNS,
        )
        feature_frame = pd.concat([pad, feature_frame], ignore_index=True)
    sequence = feature_frame.tail(seq_len_steps).to_numpy(dtype=np.float32)
    return sequence[np.newaxis, ...]


def predict_from_rows(
    bundle: ModelBundle,
    rows: list[dict[str, Any]],
    seq_len_steps: int | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bundle.model.to(device)
    model.eval()

    effective_seq_len = int(seq_len_steps or bundle.seq_len_steps)
    effective_threshold = float(bundle.threshold if threshold is None else threshold)

    sequence = rows_to_sequence(rows, effective_seq_len)

    with torch.no_grad():
        xb = torch.from_numpy(sequence).to(device)
        logits, _ = model(xb)
        probability = float(torch.sigmoid(logits).cpu().item())

    predicted_label = int(probability >= effective_threshold)
    return {
        "probability": probability,
        "threshold": effective_threshold,
        "label": predicted_label,
        "risk_level": "high" if predicted_label else "low",
        "seq_len_steps": effective_seq_len,
        "rows_used": int(sequence.shape[1]),
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "checkpoint_path": str(bundle.checkpoint_path),
        "demo_mode": bundle.is_dummy,
    }
