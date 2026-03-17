from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.constants import KEY_COLUMNS, MODEL_FEATURE_COLUMNS, TARGET_COLUMN
from src.data.preprocess.pipeline import preprocess_nicu_data


def build_sequences(df: pd.DataFrame, seq_len_steps: int) -> tuple[np.ndarray, np.ndarray]:
    x_list: list[np.ndarray] = []
    y_list: list[float] = []

    group_cols = [col for col in KEY_COLUMNS[:2] if col in df.columns]
    if group_cols != ["Hospital_ID", "Patient_ID"]:
        group_cols = ["Patient_ID"]

    for _, group in df.groupby(group_cols, sort=False):
        values = group[MODEL_FEATURE_COLUMNS + [TARGET_COLUMN]].to_numpy(dtype=np.float32)
        if len(values) <= seq_len_steps:
            continue
        for idx in range(seq_len_steps, len(values)):
            x_list.append(values[idx - seq_len_steps : idx, :-1])
            y_list.append(float(values[idx, -1]))

    if not x_list:
        return (
            np.empty((0, seq_len_steps, len(MODEL_FEATURE_COLUMNS)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


def summarize_numeric_columns(df: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    available = [col for col in columns if col in df.columns]
    if not available:
        return pd.DataFrame()

    summary = df[available].describe().T[["mean", "std", "min", "max"]]
    print(f"\n[{label}] feature sanity summary")
    print(summary.round(4).to_string())
    return summary


@dataclass(frozen=True)
class ClientBundle:
    name: str
    dataset: "TimeSeriesDataset"
    dataloader: DataLoader
    num_samples: int
    positive_prior: float


class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path: str | Path, seq_len_steps: int = 60) -> None:
        self.csv_path = Path(csv_path)
        self.seq_len_steps = int(seq_len_steps)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")

        raw_df = pd.read_csv(self.csv_path)
        if "Timestamp" not in raw_df.columns:
            raise ValueError(f"Expected a Timestamp column in {self.csv_path}")

        raw_df["Timestamp"] = pd.to_datetime(raw_df["Timestamp"])
        processed_df = preprocess_nicu_data(raw_df)
        self.processed_df = processed_df
        self.x, self.y = build_sequences(processed_df, seq_len_steps=self.seq_len_steps)

        if self.x.shape[-1] != len(MODEL_FEATURE_COLUMNS):
            raise ValueError(
                f"Feature dimension mismatch for {self.csv_path}: "
                f"expected {len(MODEL_FEATURE_COLUMNS)}, got {self.x.shape[-1]}"
            )

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.float32)

    @property
    def positive_prior(self) -> float:
        if len(self.y) == 0:
            return 0.0
        return float(np.mean(self.y))


def get_federated_dataloaders(
    mimic_csv_path: str | Path,
    pic_csv_path: str | Path,
    batch_size: int = 32,
    seq_len_steps: int = 60,
    shuffle: bool = True,
    print_sanity_summary: bool = True,
) -> tuple[dict[str, DataLoader], dict[str, float], dict[str, ClientBundle]]:
    print("Initializing Federated Clients...")

    mimic_dataset = TimeSeriesDataset(mimic_csv_path, seq_len_steps=seq_len_steps)
    pic_dataset = TimeSeriesDataset(pic_csv_path, seq_len_steps=seq_len_steps)

    mimic_loader = DataLoader(mimic_dataset, batch_size=batch_size, shuffle=shuffle)
    pic_loader = DataLoader(pic_dataset, batch_size=batch_size, shuffle=shuffle)

    total_samples = max(1, len(mimic_dataset) + len(pic_dataset))
    client_weights = {
        "Client_A_MIMIC": len(mimic_dataset) / total_samples,
        "Client_B_PIC": len(pic_dataset) / total_samples,
    }

    client_loaders = {
        "Client_A_MIMIC": mimic_loader,
        "Client_B_PIC": pic_loader,
    }

    client_bundles = {
        "Client_A_MIMIC": ClientBundle(
            name="Client_A_MIMIC",
            dataset=mimic_dataset,
            dataloader=mimic_loader,
            num_samples=len(mimic_dataset),
            positive_prior=mimic_dataset.positive_prior,
        ),
        "Client_B_PIC": ClientBundle(
            name="Client_B_PIC",
            dataset=pic_dataset,
            dataloader=pic_loader,
            num_samples=len(pic_dataset),
            positive_prior=pic_dataset.positive_prior,
        ),
    }

    print(f"Client A sequences: {len(mimic_dataset)} | weight={client_weights['Client_A_MIMIC']:.4f} | prior={mimic_dataset.positive_prior:.4f}")
    print(f"Client B sequences: {len(pic_dataset)} | weight={client_weights['Client_B_PIC']:.4f} | prior={pic_dataset.positive_prior:.4f}")

    if print_sanity_summary:
        sanity_columns = ["HR", "SpO2", "RR", "Temp", "WBC", "CRP", "Platelets", "Lactate"]
        summarize_numeric_columns(mimic_dataset.processed_df, sanity_columns, label="Client_A_MIMIC")
        summarize_numeric_columns(pic_dataset.processed_df, sanity_columns, label="Client_B_PIC")

    return client_loaders, client_weights, client_bundles


__all__ = [
    "ClientBundle",
    "TimeSeriesDataset",
    "build_sequences",
    "get_federated_dataloaders",
    "summarize_numeric_columns",
]