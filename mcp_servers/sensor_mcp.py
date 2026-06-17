"""
sensor-mcp — exposes sensor readings and anomaly scores to the agent.

This server wraps two things already built in earlier phases:
    - data/reader.py    → raw sensor windows
    - models/inference.py → BiLSTM anomaly scoring

It does NOT duplicate any logic. It's a thin HTTP layer so the agent
can reach this functionality as a "tool" via LangChain, instead of
importing Python modules directly — which is the whole point of MCP:
the agent treats this exactly like it would treat a real industrial
SCADA/IoT platform's API, even though today it's a local FastAPI app.

Endpoints:
    GET  /readings/{machine_id}?last_n=50   → raw sensor window
    GET  /anomaly_score/{machine_id}        → BiLSTM anomaly result
    GET  /machines                          → list of known machine IDs

Run:
    uvicorn mcp_servers.sensor_mcp:app --port 8001 --reload
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from data.reader import load_window
from data.store import get_all_machine_ids
from models.inference import get_anomaly_score

app = FastAPI(title="sensor-mcp", version="1.0.0")


class ReadingsResponse(BaseModel):
    machine_id: str
    machine_type: str
    n_samples: int
    period_start: str
    period_end: str
    readings: dict[str, list[float]]


class AnomalyResponse(BaseModel):
    machine_id: str
    anomaly_score: float
    severity: str
    sensor_flags: list[str]
    latest_values: dict[str, float]


@app.get("/machines", response_model=list[str])
def list_machines():
    """Return all machine IDs known to the system."""
    return get_all_machine_ids()


@app.get("/readings/{machine_id}", response_model=ReadingsResponse)
def get_readings(machine_id: str, last_n: int = 50):
    """
    Return the most recent `last_n` raw sensor readings for a machine.

    This is the lower-level tool — useful when the agent wants to see
    actual numbers rather than just a single anomaly score.
    """
    try:
        window = load_window(machine_id, last_n=last_n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return ReadingsResponse(
        machine_id=window.machine_id,
        machine_type=window.machine_type,
        n_samples=window.n_samples,
        period_start=window.timestamps[0].isoformat(),
        period_end=window.timestamps[-1].isoformat(),
        readings=window.readings,
    )


@app.get("/anomaly_score/{machine_id}", response_model=AnomalyResponse)
def anomaly_score(machine_id: str):
    """
    Run the trained BiLSTM on the machine's latest window and return
    a structured anomaly assessment. This is the PRIMARY tool the agent
    calls first when checking a machine's health.
    """
    try:
        result = get_anomaly_score(machine_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AnomalyResponse(**result.to_dict())


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "sensor-mcp"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
