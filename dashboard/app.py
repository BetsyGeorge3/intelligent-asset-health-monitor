"""
Intelligent Asset Health Monitor — Streamlit dashboard.

Five things this dashboard shows, in tabs:
    1. Overview     — machine health cards (status, anomaly score, last checked)
    2. Sensor data   — time-series charts per machine
    3. Agent trace   — the agent's step-by-step reasoning for each run
    4. Action log    — every work order, part order, dispatch, and alert
    5. Run agent     — manually trigger a live agent run for any machine

Run:
    streamlit run dashboard/app.py

Requires (for full functionality):
    - data/sensors.db populated (python setup_phase1.py)
    - models/saved/ containing a trained model (python models/train.py)
    - The 5 MCP servers running (python mcp_servers/run_all.py) — needed
      for the anomaly score on the Overview tab and for the "Run agent now"
      button. Sensor charts and historical action logs work without them.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.data_access import (
    list_machines, get_machine_window, get_latest_snapshot,
    get_anomaly_score_safe, get_work_orders, get_parts, get_part_orders,
    get_technicians, get_dispatches, get_alerts, list_saved_traces,
    save_trace, check_mcp_servers, SENSOR_LABELS,
)

st.set_page_config(
    page_title="Asset Health Monitor",
    page_icon="🛢️",
    layout="wide",
)

SEVERITY_COLORS = {
    "normal":   "#1DB876",
    "warning":  "#E8A33D",
    "critical": "#E5484D",
    "unknown":  "#888888",
}


# ---------------------------------------------------------------------------
# Sidebar — system status
# ---------------------------------------------------------------------------

def render_sidebar():
    st.sidebar.title("🛢️ Asset Health Monitor")
    st.sidebar.caption("Agentic AI for industrial predictive maintenance")

    st.sidebar.divider()
    st.sidebar.subheader("MCP server status")

    status = check_mcp_servers()
    all_up = all(status.values())

    for name, is_up in status.items():
        icon = "🟢" if is_up else "🔴"
        st.sidebar.markdown(f"{icon} `{name}`")

    if not all_up:
        st.sidebar.warning(
            "Some MCP servers are down. Run `python mcp_servers/run_all.py` "
            "in a separate terminal for full functionality."
        )

    st.sidebar.divider()
    machines = list_machines()
    if not machines:
        st.sidebar.error(
            "No machines found in the database. Run `python setup_phase1.py` first."
        )
    return machines, status


# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------

def render_overview(machines: list[str], server_status: dict):
    st.header("Machine health overview")

    if not machines:
        st.info("No machines to display yet.")
        return

    sensor_up = server_status.get("sensor-mcp", False)
    if not sensor_up:
        st.warning(
            "sensor-mcp is not running — anomaly scores can't be computed right now. "
            "Latest raw sensor values are still shown below."
        )

    cols = st.columns(len(machines))

    for col, machine_id in zip(cols, machines):
        with col:
            snapshot = get_latest_snapshot(machine_id)

            if sensor_up:
                result = get_anomaly_score_safe(machine_id)
                severity = result.get("severity", "unknown") if not result.get("error") else "unknown"
            else:
                result = None
                severity = "unknown"

            color = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["unknown"])

            st.markdown(
                f"""
                <div style="border:1px solid #333; border-radius:10px; padding:16px;
                            border-top:4px solid {color};">
                    <div style="font-size:13px; color:#888; text-transform:uppercase;
                                letter-spacing:0.05em;">{machine_id}</div>
                    <div style="font-size:22px; font-weight:600; color:{color};
                                text-transform:capitalize; margin:4px 0;">{severity}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if result and not result.get("error"):
                st.metric("Anomaly score", f"{result['anomaly_score']:.3f}")
                if result.get("sensor_flags"):
                    st.caption(f"⚠️ Elevated: {', '.join(result['sensor_flags'])}")
                else:
                    st.caption("No sensors flagged")

            if snapshot:
                st.caption(f"Last reading: {snapshot['timestamp']}")
                m1, m2 = st.columns(2)
                m1.metric("Vibration", f"{snapshot['vibration_rms']:.2f} mm/s")
                m2.metric("Temp", f"{snapshot['temperature_c']:.1f} °C")
                m3, m4 = st.columns(2)
                m3.metric("Pressure", f"{snapshot['pressure_bar']:.2f} bar")
                m4.metric("Current", f"{snapshot['current_amp']:.1f} A")
            else:
                st.caption("No sensor data available.")


# ---------------------------------------------------------------------------
# Tab 2 — Sensor charts
# ---------------------------------------------------------------------------

def render_sensor_charts(machines: list[str]):
    st.header("Sensor time series")

    if not machines:
        st.info("No machines to display yet.")
        return

    machine_id = st.selectbox("Machine", machines, key="sensor_chart_machine")
    n_points = st.slider("Number of recent readings to show", 20, 500, 100, step=20)

    df = get_machine_window(machine_id, last_n=n_points)
    if df.empty:
        st.info("No sensor data available for this machine.")
        return

    for sensor, label in SENSOR_LABELS.items():
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[sensor],
            mode="lines", name=label,
            line=dict(width=2),
        ))

        # Shade any rows where fault_type is set, so anomalies are visible at a glance
        fault_mask = df["fault_type"].notna()
        has_faults = bool(fault_mask.any())   # cast from numpy.bool_ — plotly rejects numpy bools
        if has_faults:
            fault_df = df[fault_mask]
            fig.add_trace(go.Scatter(
                x=fault_df["timestamp"], y=fault_df[sensor],
                mode="markers", name="Anomalous reading",
                marker=dict(color="#E5484D", size=6),
            ))

        fig.update_layout(
            title=label,
            height=280,
            margin=dict(l=40, r=20, t=40, b=20),
            showlegend=has_faults,
            template="plotly_dark",
        )
        st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Tab 3 — Agent reasoning trace
# ---------------------------------------------------------------------------

def render_trace_step(step: dict):
    with st.expander(f"**Step {step['step_number']}** — `{step['tool_name']}`", expanded=False):
        st.caption(step["timestamp"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Input**")
            st.json(step["tool_input"])
        with c2:
            st.markdown("**Result**")
            st.json(step["tool_output"])


def render_agent_trace(machines: list[str]):
    st.header("Agent reasoning trace")
    st.caption(
        "Every tool call the agent made, in order — input, output, and timing. "
        "This is what makes the agent's decisions auditable rather than a black box."
    )

    saved_traces = list_saved_traces()

    if not saved_traces:
        st.info(
            "No saved agent traces yet. Run `python agent/run_all_machines.py`, "
            "or use the **Run agent** tab to trigger a live run."
        )
        return

    machine_id = st.selectbox(
        "Machine", list(saved_traces.keys()), key="trace_machine"
    )
    trace = saved_traces[machine_id]

    if trace.get("error"):
        st.error(f"This run ended in an error: {trace['error']}")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Tool calls made", trace["n_steps"])
    c2.metric("Started", trace["started_at"][:19].replace("T", " "))
    c3.metric("Completed", (trace.get("completed_at") or "—")[:19].replace("T", " "))

    st.subheader("Step-by-step trace")
    for step in trace["steps"]:
        render_trace_step(step)

    st.subheader("Final summary")
    st.markdown(trace["final_summary"] or "_(no summary produced)_")


# ---------------------------------------------------------------------------
# Tab 4 — Action log
# ---------------------------------------------------------------------------

def render_action_log():
    st.header("Action log")
    st.caption("Every work order, parts order, technician dispatch, and alert generated so far.")

    sub_tabs = st.tabs(["Work orders", "Parts & inventory", "Technician dispatches", "Alerts"])

    with sub_tabs[0]:
        wo_df = get_work_orders()
        if wo_df.empty:
            st.info("No work orders created yet.")
        else:
            st.dataframe(wo_df, width='stretch', hide_index=True)

    with sub_tabs[1]:
        st.markdown("**Current stock levels**")
        parts_df = get_parts()
        if parts_df.empty:
            st.info("No part data available — start inventory-mcp at least once.")
        else:
            def highlight_low(row):
                return ["background-color: #4a1a1a" if row["low_stock"] else "" for _ in row]
            st.dataframe(
                parts_df.style.apply(highlight_low, axis=1),
                width='stretch', hide_index=True,
            )

        st.markdown("**Order history**")
        orders_df = get_part_orders()
        if orders_df.empty:
            st.info("No parts ordered yet.")
        else:
            st.dataframe(orders_df, width='stretch', hide_index=True)

    with sub_tabs[2]:
        st.markdown("**Technician availability**")
        tech_df = get_technicians()
        if tech_df.empty:
            st.info("No technician data available — start scheduling-mcp at least once.")
        else:
            st.dataframe(tech_df, width='stretch', hide_index=True)

        st.markdown("**Dispatch history**")
        dispatch_df = get_dispatches()
        if dispatch_df.empty:
            st.info("No technicians dispatched yet.")
        else:
            st.dataframe(dispatch_df, width='stretch', hide_index=True)

    with sub_tabs[3]:
        alerts_df = get_alerts()
        if alerts_df.empty:
            st.info("No alerts sent yet.")
        else:
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            alerts_df["_sort"] = alerts_df["priority"].map(priority_order).fillna(9)
            alerts_df = alerts_df.sort_values(["sent_at"], ascending=False).drop(columns="_sort")
            st.dataframe(alerts_df, width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Tab 5 — Run agent now
# ---------------------------------------------------------------------------

def render_run_agent(machines: list[str], server_status: dict):
    st.header("Run agent now")
    st.caption(
        "Trigger a live agent run for any machine. Requires all 5 MCP servers "
        "running and ANTHROPIC_API_KEY set in .env. This makes real API calls "
        "to Claude and may take 10-30 seconds."
    )

    all_up = all(server_status.values())
    if not all_up:
        st.error(
            "Not all MCP servers are running. Start them with "
            "`python mcp_servers/run_all.py` before running the agent."
        )
        return

    if not machines:
        st.info("No machines available.")
        return

    machine_id = st.selectbox("Machine to assess", machines, key="run_agent_machine")

    if st.button(f"▶ Run agent for {machine_id}", type="primary"):
        with st.spinner(f"Agent is assessing {machine_id} — calling Claude and the MCP tools..."):
            try:
                from agent.react_agent import run_agent
                trace = run_agent(machine_id)
                trace_dict = trace.to_dict()
                save_trace(trace_dict)

                if trace.error:
                    st.error(f"Agent run failed: {trace.error}")
                else:
                    st.success(
                        f"Done — {trace.n_steps} tool calls made. "
                        f"View the full trace in the **Agent trace** tab."
                    )
                    st.markdown("**Final summary:**")
                    st.markdown(trace.final_summary)

            except ImportError as e:
                st.error(
                    f"Couldn't import the agent — make sure langchain and "
                    f"langchain-anthropic are installed. ({e})"
                )
            except Exception as e:
                st.error(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    machines, server_status = render_sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "📈 Sensor data", "🧠 Agent trace", "📋 Action log", "▶ Run agent",
    ])

    with tab1:
        render_overview(machines, server_status)
    with tab2:
        render_sensor_charts(machines)
    with tab3:
        render_agent_trace(machines)
    with tab4:
        render_action_log()
    with tab5:
        render_run_agent(machines, server_status)


if __name__ == "__main__":
    main()
