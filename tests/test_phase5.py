"""
Phase 5 tests — dashboard data access layer + app-level smoke tests.

Two kinds of tests here:

1. TestDataAccess* — direct unit tests of dashboard/data_access.py's
   functions. These don't need Streamlit at all, just the underlying
   data layer (reusing the same data/sensors.db and mcp_servers/*.db
   used elsewhere in the project).

2. TestAppSmoke — runs the actual dashboard/app.py through Streamlit's
   official `AppTest` harness (streamlit.testing.v1), which executes
   the full script in-process without needing a browser or a bound
   port. This is the correct, lightweight way to test a Streamlit app:
   it catches real bugs (e.g. a numpy.bool_ vs Python bool mismatch
   that crashed plotly's `showlegend` property) that simple import
   tests would miss, since those only happen when the script actually
   executes top to bottom.

Run:
    cd asset_health_monitor
    pytest tests/test_phase5.py -v
"""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# data_access.py — pure data layer tests
# ---------------------------------------------------------------------------

class TestDataAccessEmptyState:
    """
    Every data_access.py read function must degrade gracefully (return an
    empty, correctly-columned DataFrame) when its underlying MCP database
    doesn't exist yet — e.g. on a fresh clone before any server has run.
    """

    def setup_method(self):
        import dashboard.data_access as da
        self.da = da
        self._original_cmms = da.CMMS_DB
        self._original_inv = da.INVENTORY_DB
        self._original_sched = da.SCHEDULING_DB
        self._original_alerts = da.ALERTS_LOG
        self._original_traces = da.TRACES_DIR

        # Point every DB/log path at a tmp dir guaranteed to NOT contain them
        self.tmp_dir = Path(tempfile.mkdtemp())
        da.CMMS_DB = self.tmp_dir / "cmms.db"
        da.INVENTORY_DB = self.tmp_dir / "inventory.db"
        da.SCHEDULING_DB = self.tmp_dir / "scheduling.db"
        da.ALERTS_LOG = self.tmp_dir / "alerts.log"
        da.TRACES_DIR = self.tmp_dir / "traces"

    def teardown_method(self):
        self.da.CMMS_DB = self._original_cmms
        self.da.INVENTORY_DB = self._original_inv
        self.da.SCHEDULING_DB = self._original_sched
        self.da.ALERTS_LOG = self._original_alerts
        self.da.TRACES_DIR = self._original_traces

    def test_get_work_orders_empty(self):
        df = self.da.get_work_orders()
        assert df.empty
        assert "machine_id" in df.columns

    def test_get_parts_empty(self):
        df = self.da.get_parts()
        assert df.empty
        assert "part_name" in df.columns

    def test_get_part_orders_empty(self):
        df = self.da.get_part_orders()
        assert df.empty

    def test_get_technicians_empty(self):
        df = self.da.get_technicians()
        assert df.empty
        assert "technician_id" in df.columns

    def test_get_dispatches_empty(self):
        df = self.da.get_dispatches()
        assert df.empty

    def test_get_alerts_empty(self):
        df = self.da.get_alerts()
        assert df.empty
        assert "machine_id" in df.columns

    def test_list_saved_traces_empty(self):
        traces = self.da.list_saved_traces()
        assert traces == {}

    def test_get_anomaly_score_safe_returns_error_dict_when_server_down(self):
        result = self.da.get_anomaly_score_safe("PUMP-01", sensor_url="http://localhost:19999")
        assert result.get("error") is True
        assert "detail" in result

    def test_check_mcp_servers_all_down(self):
        # Use a set of definitely-unused ports by monkeypatching the function's
        # hardcoded URLs isn't trivial here, so we just verify the real
        # localhost:800X ports (servers aren't running during this test class).
        status = self.da.check_mcp_servers()
        assert set(status.keys()) == {
            "sensor-mcp", "cmms-mcp", "inventory-mcp", "scheduling-mcp", "notify-mcp"
        }
        assert all(isinstance(v, bool) for v in status.values())


class TestDataAccessPopulatedState:
    """Tests against data_access.py functions once the underlying DBs have real rows."""

    def setup_method(self):
        import dashboard.data_access as da
        self.da = da
        self._original_cmms = da.CMMS_DB
        self._original_inv = da.INVENTORY_DB
        self._original_sched = da.SCHEDULING_DB
        self._original_alerts = da.ALERTS_LOG
        self._original_traces = da.TRACES_DIR

        self.tmp_dir = Path(tempfile.mkdtemp())
        da.CMMS_DB = self.tmp_dir / "cmms.db"
        da.INVENTORY_DB = self.tmp_dir / "inventory.db"
        da.SCHEDULING_DB = self.tmp_dir / "scheduling.db"
        da.ALERTS_LOG = self.tmp_dir / "alerts.log"
        da.TRACES_DIR = self.tmp_dir / "traces"

        self._seed_cmms()
        self._seed_inventory()
        self._seed_scheduling()
        self._seed_alerts()
        self._seed_trace()

    def teardown_method(self):
        self.da.CMMS_DB = self._original_cmms
        self.da.INVENTORY_DB = self._original_inv
        self.da.SCHEDULING_DB = self._original_sched
        self.da.ALERTS_LOG = self._original_alerts
        self.da.TRACES_DIR = self._original_traces

    def _seed_cmms(self):
        con = sqlite3.connect(self.da.CMMS_DB)
        con.execute("""CREATE TABLE work_orders (
            id INTEGER PRIMARY KEY, machine_id TEXT, fault_type TEXT,
            description TEXT, priority TEXT, status TEXT,
            created_at TEXT, updated_at TEXT)""")
        con.execute(
            "INSERT INTO work_orders VALUES (1, 'PUMP-01', 'bearing_wear', 'test', 'high', 'open', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')"
        )
        con.commit()
        con.close()

    def _seed_inventory(self):
        con = sqlite3.connect(self.da.INVENTORY_DB)
        con.execute("""CREATE TABLE parts (
            part_name TEXT PRIMARY KEY, description TEXT, quantity INTEGER,
            reorder_level INTEGER, unit_cost_aed REAL)""")
        con.execute("INSERT INTO parts VALUES ('bearing_6205', 'Bearing', 2, 5, 85.0)")  # low stock
        con.execute("""CREATE TABLE orders (
            id INTEGER PRIMARY KEY, part_name TEXT, quantity INTEGER,
            machine_id TEXT, ordered_at TEXT)""")
        con.execute("INSERT INTO orders VALUES (1, 'bearing_6205', 2, 'PUMP-01', '2026-01-01T00:00:00Z')")
        con.commit()
        con.close()

    def _seed_scheduling(self):
        con = sqlite3.connect(self.da.SCHEDULING_DB)
        con.execute("""CREATE TABLE technicians (
            technician_id TEXT PRIMARY KEY, name TEXT, specialty TEXT, available INTEGER)""")
        con.execute("INSERT INTO technicians VALUES ('TECH-01', 'Ahmed', 'bearing_wear', 0)")
        con.execute("""CREATE TABLE dispatches (
            id INTEGER PRIMARY KEY, technician_id TEXT, work_order_id INTEGER,
            machine_id TEXT, dispatched_at TEXT)""")
        con.execute("INSERT INTO dispatches VALUES (1, 'TECH-01', 1, 'PUMP-01', '2026-01-01T00:00:00Z')")
        con.commit()
        con.close()

    def _seed_alerts(self):
        self.da.ALERTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(self.da.ALERTS_LOG, "w") as f:
            f.write(json.dumps({
                "machine_id": "PUMP-01", "message": "test alert", "priority": "high",
                "channel": "console", "sent_at": "2026-01-01T00:00:00Z", "delivered": True,
            }) + "\n")

    def _seed_trace(self):
        self.da.TRACES_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.da.TRACES_DIR / "PUMP-01.json", "w") as f:
            json.dump({
                "machine_id": "PUMP-01", "started_at": "2026-01-01T00:00:00Z",
                "completed_at": "2026-01-01T00:01:00Z", "steps": [], "n_steps": 0,
                "tools_called": [], "final_summary": "All clear.", "error": None,
            }, f)

    def test_get_work_orders_returns_seeded_row(self):
        df = self.da.get_work_orders()
        assert len(df) == 1
        assert df.iloc[0]["machine_id"] == "PUMP-01"

    def test_get_parts_flags_low_stock(self):
        df = self.da.get_parts()
        assert df.iloc[0]["low_stock"] == True

    def test_get_part_orders_returns_seeded_row(self):
        df = self.da.get_part_orders()
        assert len(df) == 1

    def test_get_technicians_marks_unavailable_as_bool(self):
        df = self.da.get_technicians()
        assert df.iloc[0]["available"] == False
        assert df["available"].dtype == bool

    def test_get_dispatches_returns_seeded_row(self):
        df = self.da.get_dispatches()
        assert len(df) == 1
        assert df.iloc[0]["technician_id"] == "TECH-01"

    def test_get_alerts_returns_seeded_row(self):
        df = self.da.get_alerts()
        assert len(df) == 1
        assert df.iloc[0]["machine_id"] == "PUMP-01"

    def test_list_saved_traces_returns_seeded_trace(self):
        traces = self.da.list_saved_traces()
        assert "PUMP-01" in traces
        assert traces["PUMP-01"]["final_summary"] == "All clear."

    def test_save_trace_writes_correct_file(self):
        new_trace = {
            "machine_id": "MOTOR-01", "started_at": "x", "completed_at": "y",
            "steps": [], "n_steps": 0, "tools_called": [], "final_summary": "ok", "error": None,
        }
        out_path = self.da.save_trace(new_trace)
        assert out_path.exists()
        with open(out_path) as f:
            assert json.load(f)["machine_id"] == "MOTOR-01"


# ---------------------------------------------------------------------------
# app.py — Streamlit AppTest smoke tests
# ---------------------------------------------------------------------------

class TestAppSmoke:
    """
    Runs the real dashboard/app.py script through Streamlit's AppTest
    harness. This executes every render_* function for real (not mocked),
    which is what caught the numpy.bool_/plotly showlegend bug during
    development — a plain import test would have missed it since the
    bug only triggered when render_sensor_charts() actually ran with
    fault data present.
    """

    def test_app_runs_without_exception_no_data(self):
        """Fresh-clone scenario: no MCP servers running, no seeded action-log data."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(Path(__file__).parent.parent / "dashboard" / "app.py"))
        at.run(timeout=20)
        assert at.exception == [], f"App raised: {at.exception}"

    def test_app_has_five_top_level_tabs(self):
        """
        Note: at.tabs flattens BOTH the 5 top-level tabs AND the 4 nested
        sub-tabs inside "Action log" into one list (9 total) — that's
        correct AppTest behavior, not a bug. Check by label instead.
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(Path(__file__).parent.parent / "dashboard" / "app.py"))
        at.run(timeout=20)
        labels = {t.label for t in at.tabs}
        top_level_labels = {
            "📊 Overview", "📈 Sensor data", "🧠 Agent trace",
            "📋 Action log", "▶ Run agent",
        }
        assert top_level_labels.issubset(labels)

    def test_app_runs_without_exception_with_seeded_data(self, tmp_path, monkeypatch):
        """Same script, but with real rows in every underlying DB/log/trace file."""
        import dashboard.data_access as da

        monkeypatch.setattr(da, "CMMS_DB", tmp_path / "cmms.db")
        monkeypatch.setattr(da, "INVENTORY_DB", tmp_path / "inventory.db")
        monkeypatch.setattr(da, "SCHEDULING_DB", tmp_path / "scheduling.db")
        monkeypatch.setattr(da, "ALERTS_LOG", tmp_path / "alerts.log")
        monkeypatch.setattr(da, "TRACES_DIR", tmp_path / "traces")

        con = sqlite3.connect(da.CMMS_DB)
        con.execute("""CREATE TABLE work_orders (
            id INTEGER PRIMARY KEY, machine_id TEXT, fault_type TEXT,
            description TEXT, priority TEXT, status TEXT,
            created_at TEXT, updated_at TEXT)""")
        con.execute(
            "INSERT INTO work_orders VALUES (1, 'PUMP-01', 'bearing_wear', 'test', 'high', 'open', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')"
        )
        con.commit()
        con.close()

        (tmp_path / "alerts.log").write_text(json.dumps({
            "machine_id": "PUMP-01", "message": "test", "priority": "high",
            "channel": "console", "sent_at": "2026-01-01T00:00:00Z", "delivered": True,
        }) + "\n")

        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(Path(__file__).parent.parent / "dashboard" / "app.py"))
        at.run(timeout=20)
        assert at.exception == [], f"App raised: {at.exception}"
