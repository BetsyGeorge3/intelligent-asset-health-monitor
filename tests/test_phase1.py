"""
Phase 1 tests — data generation, store, and reader.

Run:
    cd asset_health_monitor
    pytest tests/test_phase1.py -v
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

class TestGenerator:
    def setup_method(self):
        from data.generate_sensor_data import generate
        self.generate = generate

    def test_returns_dataframe(self):
        df = self.generate(days=1)
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self):
        df = self.generate(days=1)
        expected = {"timestamp", "machine_id", "machine_type",
                    "vibration_rms", "temperature_c", "pressure_bar",
                    "current_amp", "fault_type"}
        assert expected.issubset(df.columns)

    def test_three_machines(self):
        df = self.generate(days=1)
        assert set(df["machine_id"].unique()) == {"PUMP-01", "COMPRESSOR-01", "MOTOR-01"}

    def test_no_negative_sensor_values(self):
        df = self.generate(days=5)
        for col in ["vibration_rms", "temperature_c", "pressure_bar", "current_amp"]:
            assert (df[col] >= 0).all(), f"{col} has negative values"

    def test_anomalies_injected(self):
        df = self.generate(days=30)
        assert df["fault_type"].notna().any(), "Expected fault rows, found none"

    def test_anomaly_sensors_elevated(self):
        df = self.generate(days=30)
        normal  = df[df["fault_type"].isna()]
        fault   = df[df["fault_type"].notna()]
        # During bearing wear on PUMP-01, vibration should be higher than normal
        pump_fault  = fault[fault["machine_id"] == "PUMP-01"]["vibration_rms"]
        pump_normal = normal[normal["machine_id"] == "PUMP-01"]["vibration_rms"]
        assert pump_fault.mean() > pump_normal.mean() * 1.5

    def test_sorted_by_timestamp(self):
        df = self.generate(days=2)
        for mid in df["machine_id"].unique():
            sub = df[df["machine_id"] == mid]["timestamp"]
            assert sub.is_monotonic_increasing

    def test_reproducible_with_seed(self):
        df1 = self.generate(days=2)
        df2 = self.generate(days=2)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Store tests  (use a temp DB so tests don't pollute data/sensors.db)
# ---------------------------------------------------------------------------

class TestStore:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        os.environ["SQLITE_PATH"] = self.tmp.name
        # Re-import to pick up patched env var
        import importlib
        import data.store as store_mod
        importlib.reload(store_mod)
        self.store = store_mod

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def _seed(self, days=5):
        from data.generate_sensor_data import generate
        df = generate(days=days)
        self.store.init_db()
        self.store.write_readings(df)
        return df

    def test_init_creates_table(self):
        self.store.init_db()
        import sqlite3
        con = sqlite3.connect(self.tmp.name)
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "sensor_readings" in tables

    def test_write_returns_row_count(self):
        from data.generate_sensor_data import generate
        df = generate(days=2)
        self.store.init_db()
        n = self.store.write_readings(df)
        assert n == len(df)

    def test_get_readings_returns_dataframe(self):
        self._seed()
        df = self.store.get_readings("PUMP-01", last_n=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    def test_get_readings_sorted_asc(self):
        self._seed()
        df = self.store.get_readings("PUMP-01", last_n=20)
        assert df["timestamp"].is_monotonic_increasing

    def test_get_all_machine_ids(self):
        self._seed()
        ids = self.store.get_all_machine_ids()
        assert set(ids) == {"PUMP-01", "COMPRESSOR-01", "MOTOR-01"}

    def test_get_latest_reading(self):
        self._seed()
        row = self.store.get_latest_reading("MOTOR-01")
        assert row is not None
        assert row["machine_id"] == "MOTOR-01"
        for sensor in ["vibration_rms", "temperature_c", "pressure_bar", "current_amp"]:
            assert sensor in row


# ---------------------------------------------------------------------------
# Reader tests
# ---------------------------------------------------------------------------

class TestReader:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        os.environ["SQLITE_PATH"] = self.tmp.name
        import importlib
        import data.store as store_mod
        importlib.reload(store_mod)
        from data.generate_sensor_data import generate
        df = generate(days=5)
        store_mod.init_db()
        store_mod.write_readings(df)

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def _window(self, mid="PUMP-01", n=50):
        import importlib
        import data.store as store_mod
        import data.reader as reader_mod
        importlib.reload(store_mod)
        importlib.reload(reader_mod)
        return reader_mod.load_window(mid, last_n=n)

    def test_window_array_shape(self):
        w = self._window(n=50)
        assert w.as_array.shape == (50, 4)

    def test_window_n_samples(self):
        w = self._window(n=30)
        assert w.n_samples == 30

    def test_latest_values_keys(self):
        w = self._window()
        lv = w.latest_values()
        assert set(lv.keys()) == {"vibration_rms", "temperature_c",
                                   "pressure_bar", "current_amp"}

    def test_summary_stats_structure(self):
        w = self._window()
        stats = w.summary_stats()
        for sensor in ["vibration_rms", "temperature_c", "pressure_bar", "current_amp"]:
            assert sensor in stats
            assert set(stats[sensor].keys()) == {"mean", "std", "min", "max"}

    def test_context_string_contains_machine_id(self):
        import importlib, data.store as s, data.reader as r
        importlib.reload(s); importlib.reload(r)
        w = r.load_window("COMPRESSOR-01", last_n=10)
        ctx = r.sensor_context_string(w)
        assert "COMPRESSOR-01" in ctx

    def test_invalid_machine_raises(self):
        import importlib, data.store as s, data.reader as r
        importlib.reload(s); importlib.reload(r)
        with pytest.raises(ValueError):
            r.load_window("DOES-NOT-EXIST")
