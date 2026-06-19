"""
Phase 4 tests — agent tools (against live MCP servers) and trace structures.

IMPORTANT: Unlike Phase 1-3 tests, the tool tests in this file require
the 5 MCP servers to actually be running (see mcp_servers/run_all.py),
since these tools make real HTTP calls — that's the whole point of the
MCP integration. Tests are skipped automatically if the servers aren't
reachable, rather than failing, so `pytest tests/` still works fine
without the servers up.

The full agentic loop (react_agent.py actually calling Claude) requires
ANTHROPIC_API_KEY and is NOT exercised here — that's an integration
test you run manually with `python agent/react_agent.py PUMP-01`,
not a unit test, since it costs real API tokens on every run.

Run (with MCP servers running):
    python mcp_servers/run_all.py &
    pytest tests/test_phase4.py -v
"""

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.trace import AgentRunTrace, ToolCallStep
from agent.prompts import build_system_prompt, SYSTEM_PROMPT


def _servers_running() -> bool:
    """Check if all 5 MCP servers respond to /health. Used to skip tool tests cleanly."""
    ports = [8001, 8002, 8003, 8004, 8005]
    for port in ports:
        try:
            r = httpx.get(f"http://localhost:{port}/health", timeout=1.0)
            if r.status_code != 200:
                return False
        except httpx.ConnectError:
            return False
    return True


requires_mcp_servers = pytest.mark.skipif(
    not _servers_running(),
    reason="MCP servers not running — start with `python mcp_servers/run_all.py`",
)


# ---------------------------------------------------------------------------
# Prompt tests — no servers or API key needed
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_build_system_prompt_inserts_machine_id(self):
        prompt = build_system_prompt("PUMP-01")
        assert "PUMP-01" in prompt

    def test_different_machine_ids_produce_different_prompts(self):
        p1 = build_system_prompt("PUMP-01")
        p2 = build_system_prompt("MOTOR-01")
        assert p1 != p2
        assert "MOTOR-01" in p2 and "MOTOR-01" not in p1.replace(p1, "")  # sanity

    def test_prompt_mentions_all_tool_categories(self):
        """Sanity check the prompt actually references the tool-driven workflow."""
        prompt = build_system_prompt("PUMP-01")
        for keyword in ["anomaly", "work order", "technician", "alert"]:
            assert keyword in prompt.lower()

    def test_raw_template_has_placeholder(self):
        assert "{machine_id}" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Trace tests — no servers or API key needed
# ---------------------------------------------------------------------------

class TestAgentRunTrace:
    def test_starts_with_no_steps(self):
        trace = AgentRunTrace(machine_id="PUMP-01", started_at="2026-01-01T00:00:00Z")
        assert trace.n_steps == 0
        assert trace.tools_called == []

    def test_add_step_increments_count(self):
        trace = AgentRunTrace(machine_id="PUMP-01", started_at="2026-01-01T00:00:00Z")
        trace.add_step("get_anomaly_score", {"machine_id": "PUMP-01"}, {"anomaly_score": 0.1})
        assert trace.n_steps == 1
        assert trace.tools_called == ["get_anomaly_score"]

    def test_step_numbers_increment_sequentially(self):
        trace = AgentRunTrace(machine_id="PUMP-01", started_at="2026-01-01T00:00:00Z")
        trace.add_step("tool_a", {}, {})
        trace.add_step("tool_b", {}, {})
        trace.add_step("tool_c", {}, {})
        assert [s.step_number for s in trace.steps] == [1, 2, 3]

    def test_mark_complete_sets_summary_and_timestamp(self):
        trace = AgentRunTrace(machine_id="PUMP-01", started_at="2026-01-01T00:00:00Z")
        trace.mark_complete("All normal, no action needed.")
        assert trace.final_summary == "All normal, no action needed."
        assert trace.completed_at is not None
        assert trace.error is None

    def test_mark_error_sets_error_and_timestamp(self):
        trace = AgentRunTrace(machine_id="PUMP-01", started_at="2026-01-01T00:00:00Z")
        trace.mark_error("Connection refused")
        assert trace.error == "Connection refused"
        assert trace.completed_at is not None

    def test_to_dict_structure(self):
        trace = AgentRunTrace(machine_id="PUMP-01", started_at="2026-01-01T00:00:00Z")
        trace.add_step("get_anomaly_score", {"machine_id": "PUMP-01"}, {"severity": "normal"})
        trace.mark_complete("Done.")
        d = trace.to_dict()
        assert d["machine_id"] == "PUMP-01"
        assert d["n_steps"] == 1
        assert d["tools_called"] == ["get_anomaly_score"]
        assert len(d["steps"]) == 1
        assert d["steps"][0]["tool_name"] == "get_anomaly_score"

    def test_pretty_print_includes_machine_id_and_steps(self):
        trace = AgentRunTrace(machine_id="PUMP-01", started_at="2026-01-01T00:00:00Z")
        trace.add_step("get_anomaly_score", {"machine_id": "PUMP-01"}, {"severity": "normal"})
        trace.mark_complete("All clear.")
        output = trace.pretty_print()
        assert "PUMP-01" in output
        assert "get_anomaly_score" in output
        assert "All clear." in output

    def test_pretty_print_shows_error_when_present(self):
        trace = AgentRunTrace(machine_id="PUMP-01", started_at="2026-01-01T00:00:00Z")
        trace.mark_error("Something broke")
        output = trace.pretty_print()
        assert "ERROR" in output
        assert "Something broke" in output


class TestToolCallStep:
    def test_to_dict_roundtrip(self):
        step = ToolCallStep(
            step_number=1,
            tool_name="get_anomaly_score",
            tool_input={"machine_id": "PUMP-01"},
            tool_output={"severity": "normal"},
        )
        d = step.to_dict()
        assert d["step_number"] == 1
        assert d["tool_name"] == "get_anomaly_score"
        assert d["tool_input"] == {"machine_id": "PUMP-01"}
        assert d["tool_output"] == {"severity": "normal"}
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# Tool tests — REQUIRE live MCP servers (skipped automatically if not running)
# ---------------------------------------------------------------------------

@requires_mcp_servers
class TestSensorTools:
    def setup_method(self):
        from agent.tools import get_anomaly_score, get_sensor_readings, list_machines
        self.get_anomaly_score = get_anomaly_score
        self.get_sensor_readings = get_sensor_readings
        self.list_machines = list_machines

    def test_list_machines_returns_three(self):
        result = self.list_machines.invoke({})
        assert set(result) == {"PUMP-01", "COMPRESSOR-01", "MOTOR-01"}

    def test_get_anomaly_score_valid_machine(self):
        result = self.get_anomaly_score.invoke({"machine_id": "PUMP-01"})
        assert "anomaly_score" in result
        assert result["severity"] in {"normal", "warning", "critical"}

    def test_get_anomaly_score_invalid_machine_returns_error_dict(self):
        """Tools must surface errors as data, not raise — the agent needs to see them."""
        result = self.get_anomaly_score.invoke({"machine_id": "NOT-REAL"})
        assert result.get("error") is True
        assert result["status_code"] == 404

    def test_get_sensor_readings_respects_last_n(self):
        result = self.get_sensor_readings.invoke({"machine_id": "PUMP-01", "last_n": 5})
        assert result["n_samples"] == 5


@requires_mcp_servers
class TestCMMSTools:
    def setup_method(self):
        from agent.tools import create_work_order, list_work_orders
        self.create_work_order = create_work_order
        self.list_work_orders = list_work_orders

    def test_create_work_order_returns_open_status(self):
        result = self.create_work_order.invoke({
            "machine_id": "PUMP-01", "fault_type": "bearing_wear",
            "description": "Phase 4 test work order", "priority": "high",
        })
        assert result["status"] == "open"
        assert result["priority"] == "high"

    def test_list_work_orders_filters_by_machine(self):
        self.create_work_order.invoke({
            "machine_id": "MOTOR-01", "fault_type": "overheating", "description": "test",
        })
        result = self.list_work_orders.invoke({"machine_id": "MOTOR-01"})
        assert all(wo["machine_id"] == "MOTOR-01" for wo in result)


@requires_mcp_servers
class TestInventoryTools:
    def setup_method(self):
        from agent.tools import check_part_stock, order_part
        self.check_part_stock = check_part_stock
        self.order_part = order_part

    def test_check_stock_returns_quantity(self):
        result = self.check_part_stock.invoke({"part_name": "bearing_6205"})
        assert "quantity" in result

    def test_order_exceeding_stock_returns_error_dict_not_exception(self):
        result = self.order_part.invoke({"part_name": "motor_winding_kit", "quantity": 99999})
        assert result.get("error") is True
        assert result["status_code"] == 409


@requires_mcp_servers
class TestSchedulingTools:
    def setup_method(self):
        from agent.tools import find_available_technician
        self.find_available_technician = find_available_technician

    def test_specialty_match_ranked_first(self):
        result = self.find_available_technician.invoke({"fault_type": "valve_leak"})
        assert result[0]["specialty"] == "valve_leak"


@requires_mcp_servers
class TestNotifyTools:
    def setup_method(self):
        from agent.tools import send_alert
        self.send_alert = send_alert

    def test_send_alert_returns_delivered(self):
        result = self.send_alert.invoke({
            "machine_id": "PUMP-01", "message": "Phase 4 test alert", "priority": "low",
        })
        assert result["delivered"] is True
