"""
Phase 3 tests — all 5 MCP servers.

Uses FastAPI's TestClient as a context manager (`with TestClient(app) as client`)
so that @app.on_event("startup") handlers fire correctly — this is what
actually initialises each server's SQLite tables and seed data.

Each test class uses a fresh temp SQLite file so tests never collide with
each other or with any real dev databases sitting in mcp_servers/.

Run:
    cd asset_health_monitor
    pytest tests/test_phase3.py -v
"""

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# sensor-mcp
# ---------------------------------------------------------------------------

class TestSensorMCP:
    """
    sensor-mcp wraps reader.py + inference.py, both of which depend on
    the real data/sensors.db and the trained model in models/saved/.
    These tests assume Phase 1 + Phase 2 setup has already been run
    (i.e. setup_phase1.py and models/train.py have produced real data
    and a real trained model) — this mirrors how the server is actually
    used in the project, rather than mocking those layers out.
    """

    def setup_method(self):
        # Reload data.store / data.reader / models.inference / mcp_servers.sensor_mcp
        # in case an earlier test module (e.g. test_phase1.py) patched
        # SQLITE_PATH and left a stale module reference behind.
        import importlib
        import data.store as store_mod
        import data.reader as reader_mod
        import models.inference as inference_mod
        importlib.reload(store_mod)
        importlib.reload(reader_mod)
        importlib.reload(inference_mod)

        import mcp_servers.sensor_mcp as sensor_mod
        importlib.reload(sensor_mod)
        self.client = TestClient(sensor_mod.app)

    def test_health(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        assert r.json()["service"] == "sensor-mcp"

    def test_list_machines(self):
        r = self.client.get("/machines")
        assert r.status_code == 200
        assert set(r.json()) == {"PUMP-01", "COMPRESSOR-01", "MOTOR-01"}

    def test_get_readings_valid_machine(self):
        r = self.client.get("/readings/PUMP-01", params={"last_n": 10})
        assert r.status_code == 200
        body = r.json()
        assert body["machine_id"] == "PUMP-01"
        assert body["n_samples"] == 10
        assert set(body["readings"].keys()) == {
            "vibration_rms", "temperature_c", "pressure_bar", "current_amp"
        }

    def test_get_readings_invalid_machine_404(self):
        r = self.client.get("/readings/NOT-A-MACHINE")
        assert r.status_code == 404

    def test_anomaly_score_valid_machine(self):
        r = self.client.get("/anomaly_score/PUMP-01")
        assert r.status_code == 200
        body = r.json()
        assert body["machine_id"] == "PUMP-01"
        assert 0.0 <= body["anomaly_score"] <= 1.0
        assert body["severity"] in {"normal", "warning", "critical"}

    def test_anomaly_score_invalid_machine_404(self):
        r = self.client.get("/anomaly_score/NOT-A-MACHINE")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# cmms-mcp
# ---------------------------------------------------------------------------

class TestCMMSMCP:
    def setup_method(self):
        # Point this server at a fresh temp DB so tests don't pollute
        # mcp_servers/cmms.db or collide with other test runs.
        self.tmp_dir = tempfile.mkdtemp()
        import mcp_servers.cmms_mcp as cmms_mod
        importlib.reload(cmms_mod)
        cmms_mod.DB_PATH = Path(self.tmp_dir) / "cmms.db"
        self.app = cmms_mod.app
        self.client_cm = TestClient(self.app)

    def test_create_and_get_work_order(self):
        with self.client_cm as client:
            r = client.post("/work_orders", json={
                "machine_id": "PUMP-01",
                "fault_type": "bearing_wear",
                "description": "Test work order",
                "priority": "high",
            })
            assert r.status_code == 200
            wo = r.json()
            assert wo["status"] == "open"
            assert wo["machine_id"] == "PUMP-01"

            r2 = client.get(f"/work_orders/{wo['id']}")
            assert r2.status_code == 200
            assert r2.json()["id"] == wo["id"]

    def test_default_priority_is_medium(self):
        with self.client_cm as client:
            r = client.post("/work_orders", json={
                "machine_id": "MOTOR-01", "fault_type": "overheating", "description": "test"
            })
            assert r.json()["priority"] == "medium"

    def test_list_filters_by_machine_id(self):
        with self.client_cm as client:
            client.post("/work_orders", json={
                "machine_id": "PUMP-01", "fault_type": "bearing_wear", "description": "a"
            })
            client.post("/work_orders", json={
                "machine_id": "MOTOR-01", "fault_type": "overheating", "description": "b"
            })
            r = client.get("/work_orders", params={"machine_id": "PUMP-01"})
            results = r.json()
            assert len(results) == 1
            assert results[0]["machine_id"] == "PUMP-01"

    def test_status_update(self):
        with self.client_cm as client:
            wo = client.post("/work_orders", json={
                "machine_id": "PUMP-01", "fault_type": "bearing_wear", "description": "test"
            }).json()
            r = client.patch(f"/work_orders/{wo['id']}/status", json={"status": "completed"})
            assert r.status_code == 200
            assert r.json()["status"] == "completed"

    def test_get_nonexistent_work_order_404(self):
        with self.client_cm as client:
            r = client.get("/work_orders/99999")
            assert r.status_code == 404

    def test_update_nonexistent_work_order_404(self):
        with self.client_cm as client:
            r = client.patch("/work_orders/99999/status", json={"status": "completed"})
            assert r.status_code == 404


# ---------------------------------------------------------------------------
# inventory-mcp
# ---------------------------------------------------------------------------

class TestInventoryMCP:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        import mcp_servers.inventory_mcp as inv_mod
        importlib.reload(inv_mod)
        inv_mod.DB_PATH = Path(self.tmp_dir) / "inventory.db"
        self.client_cm = TestClient(inv_mod.app)

    def test_list_parts_seeded(self):
        with self.client_cm as client:
            r = client.get("/parts")
            assert r.status_code == 200
            assert len(r.json()) == 8   # matches SEED_PARTS count

    def test_get_specific_part(self):
        with self.client_cm as client:
            r = client.get("/parts/bearing_6205")
            assert r.status_code == 200
            assert r.json()["quantity"] == 12

    def test_get_nonexistent_part_404(self):
        with self.client_cm as client:
            r = client.get("/parts/not_a_real_part")
            assert r.status_code == 404

    def test_order_decrements_stock(self):
        with self.client_cm as client:
            before = client.get("/parts/bearing_6205").json()["quantity"]
            client.post("/parts/order", json={"part_name": "bearing_6205", "quantity": 3})
            after = client.get("/parts/bearing_6205").json()["quantity"]
            assert after == before - 3

    def test_order_exceeding_stock_returns_409(self):
        with self.client_cm as client:
            r = client.post("/parts/order", json={"part_name": "motor_winding_kit", "quantity": 999})
            assert r.status_code == 409

    def test_order_nonexistent_part_404(self):
        with self.client_cm as client:
            r = client.post("/parts/order", json={"part_name": "fake_part", "quantity": 1})
            assert r.status_code == 404

    def test_low_stock_flag(self):
        with self.client_cm as client:
            # motor_winding_kit seeded at qty=2, reorder_level=1 → not low yet
            r = client.get("/parts/motor_winding_kit")
            assert r.json()["low_stock"] is False
            # Order 1, leaving qty=1 == reorder_level=1 → now low
            client.post("/parts/order", json={"part_name": "motor_winding_kit", "quantity": 1})
            r = client.get("/parts/motor_winding_kit")
            assert r.json()["low_stock"] is True


# ---------------------------------------------------------------------------
# scheduling-mcp
# ---------------------------------------------------------------------------

class TestSchedulingMCP:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        import mcp_servers.scheduling_mcp as sched_mod
        importlib.reload(sched_mod)
        sched_mod.DB_PATH = Path(self.tmp_dir) / "scheduling.db"
        self.client_cm = TestClient(sched_mod.app)

    def test_list_technicians_seeded(self):
        with self.client_cm as client:
            r = client.get("/technicians")
            assert len(r.json()) == 4

    def test_available_filters_by_specialty_with_exact_match_first(self):
        with self.client_cm as client:
            r = client.get("/technicians/available", params={"fault_type": "valve_leak"})
            results = r.json()
            assert results[0]["technician_id"] == "TECH-02"   # exact specialty match

    def test_general_fallback_included(self):
        with self.client_cm as client:
            r = client.get("/technicians/available", params={"fault_type": "valve_leak"})
            ids = [t["technician_id"] for t in r.json()]
            assert "TECH-04" in ids   # general fallback technician included

    def test_dispatch_marks_unavailable(self):
        with self.client_cm as client:
            client.post("/dispatch", json={"technician_id": "TECH-01", "machine_id": "PUMP-01"})
            r = client.get("/technicians")
            tech01 = next(t for t in r.json() if t["technician_id"] == "TECH-01")
            assert tech01["available"] is False

    def test_dispatch_already_unavailable_409(self):
        with self.client_cm as client:
            client.post("/dispatch", json={"technician_id": "TECH-01", "machine_id": "PUMP-01"})
            r = client.post("/dispatch", json={"technician_id": "TECH-01", "machine_id": "MOTOR-01"})
            assert r.status_code == 409

    def test_dispatch_nonexistent_technician_404(self):
        with self.client_cm as client:
            r = client.post("/dispatch", json={"technician_id": "TECH-99", "machine_id": "PUMP-01"})
            assert r.status_code == 404


# ---------------------------------------------------------------------------
# notify-mcp
# ---------------------------------------------------------------------------

class TestNotifyMCP:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        import mcp_servers.notify_mcp as notify_mod
        importlib.reload(notify_mod)
        notify_mod.LOG_PATH = Path(self.tmp_dir) / "alerts.log"
        self.client_cm = TestClient(notify_mod.app)

    def test_send_alert_returns_delivered_true(self):
        with self.client_cm as client:
            r = client.post("/alert", json={
                "machine_id": "PUMP-01", "message": "test alert", "priority": "high"
            })
            assert r.status_code == 200
            assert r.json()["delivered"] is True

    def test_default_priority_and_channel(self):
        with self.client_cm as client:
            r = client.post("/alert", json={"machine_id": "PUMP-01", "message": "test"})
            body = r.json()
            assert body["priority"] == "medium"
            assert body["channel"] == "console"

    def test_alerts_appear_in_history(self):
        with self.client_cm as client:
            client.post("/alert", json={"machine_id": "PUMP-01", "message": "alert 1"})
            client.post("/alert", json={"machine_id": "MOTOR-01", "message": "alert 2"})
            r = client.get("/alerts")
            assert len(r.json()) == 2

    def test_alerts_filtered_by_machine_id(self):
        with self.client_cm as client:
            client.post("/alert", json={"machine_id": "PUMP-01", "message": "alert 1"})
            client.post("/alert", json={"machine_id": "MOTOR-01", "message": "alert 2"})
            r = client.get("/alerts", params={"machine_id": "PUMP-01"})
            results = r.json()
            assert len(results) == 1
            assert results[0]["machine_id"] == "PUMP-01"
