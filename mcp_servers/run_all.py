"""
Launch all 5 MCP servers as background processes for local development.

This is a convenience script — in production each server would be its
own container/Lambda, but for local dev and demos it's much easier to
start everything with one command and stop it with Ctrl+C.

Usage:
    cd asset_health_monitor
    python mcp_servers/run_all.py
"""

import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

SERVERS = [
    ("sensor-mcp",     "mcp_servers.sensor_mcp:app",     8001),
    ("cmms-mcp",       "mcp_servers.cmms_mcp:app",       8002),
    ("inventory-mcp",  "mcp_servers.inventory_mcp:app",  8003),
    ("scheduling-mcp", "mcp_servers.scheduling_mcp:app", 8004),
    ("notify-mcp",     "mcp_servers.notify_mcp:app",     8005),
]

processes = []


def start_all():
    for name, module_path, port in SERVERS:
        print(f"Starting {name} on port {port}...")
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", module_path,
             "--host", "0.0.0.0", "--port", str(port)],
            cwd=ROOT,
        )
        processes.append((name, proc))
    time.sleep(2)
    print("\nAll MCP servers running:")
    for name, _, port in SERVERS:
        print(f"  {name:<16} http://localhost:{port}/docs")
    print("\nPress Ctrl+C to stop all servers.\n")


def stop_all(*_):
    print("\nStopping all MCP servers...")
    for name, proc in processes:
        proc.terminate()
    for name, proc in processes:
        proc.wait()
    print("All stopped.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)
    start_all()
    # Keep the main process alive while children run
    while True:
        time.sleep(1)
