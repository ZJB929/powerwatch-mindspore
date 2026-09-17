from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Series:
    timestamps: list[str]
    values: np.ndarray
    labels: np.ndarray


def load_series(path: Path) -> Series:
    timestamps: list[str] = []
    values: list[float] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            timestamps.append(row["timestamp"])
            values.append(float(row["power_kw"]))
            labels.append(int(row.get("label", "0")))
    if not values:
        raise ValueError("empty series")
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("power_kw must contain finite non-negative values")
    return Series(timestamps, array, np.asarray(labels, dtype=np.int32))


def split_points(length: int, train_ratio: float = 0.6, valid_ratio: float = 0.2) -> tuple[int, int]:
    if length < 10:
        raise ValueError("series is too short")
    train_end = int(length * train_ratio)
    valid_end = int(length * (train_ratio + valid_ratio))
    return train_end, valid_end


def fit_scaler(train_values: np.ndarray) -> tuple[float, float]:
    mean = float(train_values.mean())
    std = float(train_values.std())
    return mean, max(std, 1e-6)


def make_windows(
    values: np.ndarray,
    labels: np.ndarray,
    window: int,
    stride: int,
    mean: float,
    std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window <= 0 or stride <= 0 or len(values) < window:
        raise ValueError("invalid window or stride")
    xs: list[np.ndarray] = []
    ys: list[int] = []
    ends: list[int] = []
    for start in range(0, len(values) - window + 1, stride):
        end = start + window
        xs.append((values[start:end] - mean) / std)
        ys.append(int(labels[start:end].max()))
        ends.append(end - 1)
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.int32),
        np.asarray(ends, dtype=np.int64),
    )
