"""
Run the agent across all monitored machines in one go, saving each
run's trace to disk as JSON — useful for demos and for feeding the
Streamlit dashboard (Phase 5) without needing the dashboard to invoke
the agent live.

Requires:
    - ANTHROPIC_API_KEY set in .env
    - All 5 MCP servers running (python mcp_servers/run_all.py)

Usage:
    python agent/run_all_machines.py
    python agent/run_all_machines.py --output-dir agent/traces
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.store import get_all_machine_ids
from agent.react_agent import run_agent


def main():
    parser = argparse.ArgumentParser(description="Run the agent across all machines")
    parser.add_argument("--output-dir", type=str, default="agent/traces",
                         help="Directory to save trace JSON files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    machine_ids = get_all_machine_ids()
    print(f"Running agent across {len(machine_ids)} machines: {machine_ids}\n")

    for machine_id in machine_ids:
        print(f"{'='*70}\nAssessing {machine_id}...\n{'='*70}")
        trace = run_agent(machine_id)
        print(trace.pretty_print())

        out_path = out_dir / f"{machine_id}.json"
        with open(out_path, "w") as f:
            json.dump(trace.to_dict(), f, indent=2)
        print(f"\nSaved trace -> {out_path}\n")

    print("All machines assessed.")


if __name__ == "__main__":
    main()
