"""
Phase 2 tests — BiLSTM architecture, dataset windowing, and inference.

Run:
    cd asset_health_monitor
    pytest tests/test_phase2.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bilstm import BiLSTMAnomalyDetector
from models.dataset import build_windows, train_val_test_split, Normalizer, SensorWindowDataset


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

class TestBiLSTMArchitecture:
    def test_output_shape_matches_batch_size(self):
        model = BiLSTMAnomalyDetector(n_sensors=4)
        x = torch.randn(16, 50, 4)
        out = model(x)
        assert out.shape == (16,)

    def test_output_in_valid_probability_range(self):
        model = BiLSTMAnomalyDetector(n_sensors=4)
        x = torch.randn(8, 50, 4)
        out = model(x)
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0)

    def test_handles_different_window_sizes(self):
        model = BiLSTMAnomalyDetector(n_sensors=4)
        for seq_len in [10, 50, 100]:
            x = torch.randn(4, seq_len, 4)
            out = model(x)
            assert out.shape == (4,)

    def test_predict_score_single_sample(self):
        model = BiLSTMAnomalyDetector(n_sensors=4)
        x = torch.randn(50, 4)   # no batch dim
        score = model.predict_score(x)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_different_inputs_give_different_outputs(self):
        """Sanity check the model isn't collapsed to a constant output."""
        torch.manual_seed(0)
        model = BiLSTMAnomalyDetector(n_sensors=4)
        x1 = torch.randn(1, 50, 4)
        x2 = torch.randn(1, 50, 4) * 10   # very different scale
        out1 = model(x1).item()
        out2 = model(x2).item()
        assert out1 != out2

    def test_param_count_reasonable(self):
        """Guards against accidentally creating a massive or empty model."""
        model = BiLSTMAnomalyDetector(n_sensors=4, hidden_size=64, num_layers=2)
        n_params = sum(p.numel() for p in model.parameters())
        assert 10_000 < n_params < 1_000_000


# ---------------------------------------------------------------------------
# Dataset / windowing tests
# ---------------------------------------------------------------------------

class TestBuildWindows:
    def _toy_df(self, n_rows=100, machine_id="M1", fault_start=40, fault_end=60):
        """Small synthetic DataFrame for fast, deterministic window tests."""
        rows = []
        for i in range(n_rows):
            fault = "test_fault" if fault_start <= i < fault_end else None
            rows.append({
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
                "machine_id": machine_id,
                "machine_type": "test_type",
                "vibration_rms": 1.0,
                "temperature_c": 50.0,
                "pressure_bar": 5.0,
                "current_amp": 10.0,
                "fault_type": fault,
            })
        return pd.DataFrame(rows)

    def test_window_shape(self):
        df = self._toy_df(n_rows=100)
        X, y, mids = build_windows(df, window_size=50, stride=5)
        assert X.shape[1:] == (50, 4)   # (window_size, n_sensors)

    def test_correct_number_of_windows(self):
        df = self._toy_df(n_rows=100)
        # windows start at 0, 5, 10, ... while start+50 <= 100 → starts 0..50 step 5 → 11 windows
        X, y, mids = build_windows(df, window_size=50, stride=5)
        assert len(y) == 11

    def test_window_labeled_anomalous_if_any_fault_inside(self):
        df = self._toy_df(n_rows=100, fault_start=40, fault_end=60)
        X, y, mids = build_windows(df, window_size=50, stride=5)
        # A window starting at 0 (covers rows 0-49) touches fault rows 40-49 → should be positive
        assert y[0] == 1.0

    def test_window_labeled_normal_if_no_fault(self):
        df = self._toy_df(n_rows=100, fault_start=200, fault_end=210)  # fault outside range
        X, y, mids = build_windows(df, window_size=50, stride=5)
        assert (y == 0.0).all()

    def test_machine_ids_returned_per_window(self):
        df = self._toy_df(n_rows=100, machine_id="PUMP-99")
        X, y, mids = build_windows(df, window_size=50, stride=5)
        assert all(m == "PUMP-99" for m in mids)

    def test_multiple_machines_handled_independently(self):
        df1 = self._toy_df(n_rows=100, machine_id="M1", fault_start=40, fault_end=60)
        df2 = self._toy_df(n_rows=100, machine_id="M2", fault_start=200, fault_end=210)
        df = pd.concat([df1, df2], ignore_index=True)
        X, y, mids = build_windows(df, window_size=50, stride=5)
        # Each machine independently produces 11 windows → 22 total
        assert len(y) == 22
        assert set(mids) == {"M1", "M2"}


class TestTrainValTestSplit:
    def _toy_xy(self, n=1000, pos_rate=0.1):
        X = np.random.randn(n, 50, 4).astype(np.float32)
        n_pos = int(n * pos_rate)
        y = np.array([1.0] * n_pos + [0.0] * (n - n_pos), dtype=np.float32)
        mids = ["M1"] * n
        return X, y, mids

    def test_splits_are_disjoint_and_complete(self):
        X, y, mids = self._toy_xy(n=1000)
        splits = train_val_test_split(X, y, mids)
        total = sum(len(v[1]) for v in splits.values())
        assert total == 1000   # no rows lost or duplicated across splits

    def test_stratification_preserves_positive_rate(self):
        X, y, mids = self._toy_xy(n=2000, pos_rate=0.1)
        splits = train_val_test_split(X, y, mids)
        for name, (Xs, ys) in splits.items():
            rate = ys.mean()
            assert abs(rate - 0.1) < 0.03, f"{name} split positive rate drifted too far: {rate}"

    def test_val_and_test_fractions_roughly_correct(self):
        X, y, mids = self._toy_xy(n=1000)
        splits = train_val_test_split(X, y, mids, val_frac=0.15, test_frac=0.15)
        assert abs(len(splits["val"][1])  - 150) <= 5
        assert abs(len(splits["test"][1]) - 150) <= 5
        assert abs(len(splits["train"][1]) - 700) <= 10


class TestNormalizer:
    def test_fit_produces_correct_shapes(self):
        df = pd.DataFrame({
            "vibration_rms": [1.0, 2.0, 3.0],
            "temperature_c": [50.0, 60.0, 70.0],
            "pressure_bar":  [5.0, 5.0, 5.0],
            "current_amp":   [10.0, 11.0, 12.0],
        })
        norm = Normalizer.fit(df)
        assert norm.mean.shape == (4,)
        assert norm.std.shape == (4,)

    def test_constant_sensor_does_not_divide_by_zero(self):
        """pressure_bar is constant (std=0) — must not produce NaN/inf."""
        df = pd.DataFrame({
            "vibration_rms": [1.0, 2.0, 3.0],
            "temperature_c": [50.0, 60.0, 70.0],
            "pressure_bar":  [5.0, 5.0, 5.0],
            "current_amp":   [10.0, 11.0, 12.0],
        })
        norm = Normalizer.fit(df)
        arr = df[["vibration_rms", "temperature_c", "pressure_bar", "current_amp"]].values
        transformed = norm.transform(arr)
        assert not np.isnan(transformed).any()
        assert not np.isinf(transformed).any()

    def test_transform_roughly_zero_mean_unit_std(self):
        rng = np.random.default_rng(0)
        data = rng.normal(loc=[2, 50, 5, 10], scale=[0.5, 2, 0.3, 1], size=(1000, 4))
        df = pd.DataFrame(data, columns=["vibration_rms", "temperature_c", "pressure_bar", "current_amp"])
        norm = Normalizer.fit(df)
        transformed = norm.transform(data)
        assert np.allclose(transformed.mean(axis=0), 0, atol=0.1)
        assert np.allclose(transformed.std(axis=0), 1, atol=0.1)

    def test_roundtrip_dict_serialisation(self):
        df = pd.DataFrame({
            "vibration_rms": [1.0, 2.0], "temperature_c": [50.0, 60.0],
            "pressure_bar": [5.0, 6.0], "current_amp": [10.0, 11.0],
        })
        norm = Normalizer.fit(df)
        restored = Normalizer.from_dict(norm.to_dict())
        assert np.allclose(norm.mean, restored.mean)
        assert np.allclose(norm.std, restored.std)


class TestSensorWindowDataset:
    def test_len_matches_input(self):
        X = np.random.randn(20, 50, 4).astype(np.float32)
        y = np.zeros(20, dtype=np.float32)
        norm = Normalizer(mean=np.zeros(4, dtype=np.float32), std=np.ones(4, dtype=np.float32))
        ds = SensorWindowDataset(X, y, norm)
        assert len(ds) == 20

    def test_getitem_returns_tensors(self):
        X = np.random.randn(5, 50, 4).astype(np.float32)
        y = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        norm = Normalizer(mean=np.zeros(4, dtype=np.float32), std=np.ones(4, dtype=np.float32))
        ds = SensorWindowDataset(X, y, norm)
        x_item, y_item = ds[0]
        assert isinstance(x_item, torch.Tensor)
        assert isinstance(y_item, torch.Tensor)
        assert x_item.shape == (50, 4)
