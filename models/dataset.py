"""
Dataset builder for the BiLSTM anomaly detector.

Takes the flat, row-per-timestamp DataFrame from generate_sensor_data.py
and slices it into overlapping fixed-length windows, each labeled 0 (normal)
or 1 (anomalous), ready to feed into PyTorch.

Windowing strategy:
    - window_size=50 readings (~4 hours at 5-min sampling) — matches the
      same window size used everywhere else in the project (reader.py,
      the agent's MCP sensor calls), so the model sees the same shape
      of data at train time and inference time.
    - stride=5 — slide the window forward 5 readings at a time, which
      gives meaningful overlap without exploding dataset size.
    - A window is labeled anomalous (1) if ANY reading inside it has a
      non-null fault_type. This means the model learns to flag a window
      as soon as a fault starts appearing anywhere in it, not just when
      the window is 100% fault — this is what gives early warning value.

Also includes feature normalisation (z-score per sensor, fit on the
training split only, to avoid leaking test statistics into training).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SENSOR_COLS = ["vibration_rms", "temperature_c", "pressure_bar", "current_amp"]


@dataclass
class Normalizer:
    """Per-sensor z-score normalisation. Fit once on training data, reused everywhere."""
    mean: np.ndarray   # shape (n_sensors,)
    std:  np.ndarray   # shape (n_sensors,)

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Normalizer":
        mean = df[SENSOR_COLS].mean().values.astype(np.float32)
        std  = df[SENSOR_COLS].std().values.astype(np.float32)
        std  = np.where(std < 1e-6, 1.0, std)   # avoid divide-by-zero on a constant sensor
        return cls(mean=mean, std=std)

    def transform(self, arr: np.ndarray) -> np.ndarray:
        """arr: (..., n_sensors) → normalised array, same shape."""
        return (arr - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(mean=np.array(d["mean"], dtype=np.float32),
                    std=np.array(d["std"], dtype=np.float32))


def build_windows(
    df: pd.DataFrame,
    window_size: int = 50,
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Slice a flat sensor DataFrame into overlapping windows, per machine.

    Args:
        df: DataFrame with columns timestamp, machine_id, SENSOR_COLS, fault_type.
            Must already be sorted by timestamp within each machine_id.
        window_size: number of consecutive readings per window.
        stride: step size between window start points.

    Returns:
        X:          (n_windows, window_size, n_sensors) float32 array
        y:          (n_windows,) float32 array — 1.0 if any reading in the
                    window is anomalous, else 0.0
        machine_ids: list of machine_id per window (for stratified splitting)
    """
    X_list, y_list, mid_list = [], [], []

    for machine_id, group in df.groupby("machine_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        sensor_arr = group[SENSOR_COLS].values.astype(np.float32)
        fault_arr  = group["fault_type"].notna().values   # bool array

        n = len(group)
        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            X_list.append(sensor_arr[start:end])
            y_list.append(1.0 if fault_arr[start:end].any() else 0.0)
            mid_list.append(machine_id)

    X = np.stack(X_list)                       # (n_windows, window_size, n_sensors)
    y = np.array(y_list, dtype=np.float32)      # (n_windows,)
    return X, y, mid_list


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    machine_ids: list[str],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Split windows into train/val/test, stratified by label so each split
    has a similar anomaly ratio (important since anomalies are only ~5%
    of the data — a random split could starve one split of positives).

    Returns:
        dict with keys "train", "val", "test", each mapping to (X, y) tuples.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)

    # Stratify: split anomalous and normal indices separately, then combine
    pos_idx = idx[y == 1.0]
    neg_idx = idx[y == 0.0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def split_indices(arr):
        n_test = int(len(arr) * test_frac)
        n_val  = int(len(arr) * val_frac)
        test  = arr[:n_test]
        val   = arr[n_test:n_test + n_val]
        train = arr[n_test + n_val:]
        return train, val, test

    pos_train, pos_val, pos_test = split_indices(pos_idx)
    neg_train, neg_val, neg_test = split_indices(neg_idx)

    train_idx = np.concatenate([pos_train, neg_train])
    val_idx   = np.concatenate([pos_val,   neg_val])
    test_idx  = np.concatenate([pos_test,  neg_test])

    rng.shuffle(train_idx)   # shuffle within split (order doesn't matter for training)

    return {
        "train": (X[train_idx], y[train_idx]),
        "val":   (X[val_idx],   y[val_idx]),
        "test":  (X[test_idx],  y[test_idx]),
    }


class SensorWindowDataset(Dataset):
    """PyTorch Dataset wrapping pre-built (X, y) window arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray, normalizer: Normalizer):
        self.X = normalizer.transform(X).astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.generate_sensor_data import generate

    print("Generating 30 days of synthetic data...")
    df = generate(days=30)

    print("Building windows (size=50, stride=5)...")
    X, y, mids = build_windows(df, window_size=50, stride=5)
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}, positive rate: {y.mean()*100:.2f}%")

    print("\nSplitting train/val/test (stratified)...")
    splits = train_val_test_split(X, y, mids)
    for name, (Xs, ys) in splits.items():
        print(f"  {name:<6} n={len(ys):<6} positive_rate={ys.mean()*100:.2f}%")
