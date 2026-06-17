"""
notify-mcp — mock alerting system, standing in for Twilio (SMS),
SendGrid (email), or a Slack webhook.

This is intentionally the simplest server: it logs alerts to a file
and prints them to console. In a real deployment, swap the body of
`send_alert()` for an actual Twilio/SendGrid/Slack API call — the
agent-facing interface (POST /alert) wouldn't need to change at all.

Endpoints:
    POST /alert        → send an alert
    GET  /alerts        → view alert history (useful for the dashboard)

Run:
    uvicorn mcp_servers.notify_mcp:app --port 8005 --reload
"""

import json
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcp_servers.schemas import Priority, now_iso

LOG_PATH = Path(__file__).parent / "alerts.log"

app = FastAPI(title="notify-mcp", version="1.0.0")


class AlertRequest(BaseModel):
    machine_id: str
    message: str
    priority: Priority = Priority.medium
    channel: str = "console"   # "console" | "sms" | "email" | "slack" — mock only


class AlertResponse(BaseModel):
    machine_id: str
    message: str
    priority: str
    channel: str
    sent_at: str
    delivered: bool


def _append_log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _read_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


@app.post("/alert", response_model=AlertResponse)
def send_alert(req: AlertRequest):
    """
    Send an alert. Currently logs to file + prints to console.
    Swap this implementation for a real Twilio/SendGrid/Slack call
    when moving from mock to production — the agent's tool call
    signature stays identical.
    """
    ts = now_iso()
    entry = {
        "machine_id": req.machine_id,
        "message": req.message,
        "priority": req.priority.value,
        "channel": req.channel,
        "sent_at": ts,
        "delivered": True,
    }

    # --- MOCK DELIVERY --------------------------------------------------
    print(f"[ALERT - {req.priority.value.upper()}] ({req.channel}) "
          f"{req.machine_id}: {req.message}")
    _append_log(entry)
    # ----------------------------------------------------------------------
    # Real implementation example (Twilio SMS):
    #   from twilio.rest import Client
    #   client = Client(account_sid, auth_token)
    #   client.messages.create(body=req.message, from_=FROM_NUMBER, to=TO_NUMBER)
    # ----------------------------------------------------------------------

    return AlertResponse(**entry)


@app.get("/alerts", response_model=list[AlertResponse])
def list_alerts(machine_id: str | None = None):
    """View alert history — useful for the Streamlit dashboard's action log."""
    alerts = _read_log()
    if machine_id:
        alerts = [a for a in alerts if a["machine_id"] == machine_id]
    return [AlertResponse(**a) for a in alerts]


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "notify-mcp"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
