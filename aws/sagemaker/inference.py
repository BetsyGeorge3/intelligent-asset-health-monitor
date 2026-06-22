"""
SageMaker inference entry point for the BiLSTM anomaly detector.

SageMaker's PyTorch inference containers look for four specific
functions in this file (the names are a SageMaker convention, not
something we chose):

    model_fn(model_dir)              — load the model once per worker
    input_fn(request_body, content_type)   — parse the incoming request
    predict_fn(input_data, model)          — run inference
    output_fn(prediction, accept)          — serialize the response

This wraps the exact same BiLSTMAnomalyDetector + Normalizer from
models/bilstm.py and models/dataset.py — no model logic is duplicated
here, only the SageMaker-specific request/response plumbing.

Packaging:
    See aws/sagemaker/package_model.py, which bundles this file +
    models/bilstm.py + models/dataset.py + the trained weights into
    the model.tar.gz SageMaker expects.

Deploying:
    See aws/sagemaker/deploy.py for the boto3/sagemaker SDK calls that
    create the model, endpoint config, and endpoint from that tarball.

Once deployed, models/inference.py's get_anomaly_score() can be pointed
at the SageMaker endpoint instead of loading local weights — see the
USE_SAGEMAKER_ENDPOINT note in that file.
"""

import json
import os

import numpy as np
import torch

# These two imports are why package_model.py bundles bilstm.py and
# dataset.py alongside this file — SageMaker's container only has
# what's inside model.tar.gz, not the rest of this repo.
from bilstm import BiLSTMAnomalyDetector
from dataset import Normalizer, SENSOR_COLS


def model_fn(model_dir: str):
    """
    Called once when the SageMaker endpoint's container starts.

    Args:
        model_dir: path SageMaker extracts model.tar.gz into (e.g. /opt/ml/model)

    Returns:
        A dict bundling the loaded model + normalizer + window size —
        passed as-is into predict_fn on every request.
    """
    with open(os.path.join(model_dir, "model_metadata.json")) as f:
        metadata = json.load(f)

    cfg = metadata["model_config"]
    model = BiLSTMAnomalyDetector(
        n_sensors=cfg["n_sensors"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
    )
    weights_path = os.path.join(model_dir, "bilstm_weights.pt")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    normalizer = Normalizer.from_dict(metadata["normalizer"])

    return {
        "model": model,
        "normalizer": normalizer,
        "window_size": cfg["window_size"],
    }


def input_fn(request_body: str, content_type: str = "application/json"):
    """
    Parse the incoming inference request.

    Expected JSON body:
        {
            "readings": {
                "vibration_rms":  [<window_size floats>],
                "temperature_c":  [<window_size floats>],
                "pressure_bar":   [<window_size floats>],
                "current_amp":    [<window_size floats>]
            }
        }

    This mirrors SensorWindow.readings from data/reader.py — the
    caller (sensor-mcp, or anything else) is expected to have already
    pulled the raw window via data.reader.load_window() and just pass
    the `readings` dict through.
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    payload = json.loads(request_body)
    readings = payload["readings"]

    # Build (window_size, n_sensors) array in the fixed SENSOR_COLS order —
    # same convention as data/reader.py's SensorWindow.as_array
    arr = np.array([readings[s] for s in SENSOR_COLS], dtype=np.float32).T
    return arr


def predict_fn(input_data: np.ndarray, model_bundle: dict) -> dict:
    """
    Run the BiLSTM on a parsed (window_size, n_sensors) array.

    Returns a plain dict (not yet JSON-serialized) — output_fn handles
    that — so this function's logic is testable without going through
    string serialization.
    """
    model = model_bundle["model"]
    normalizer = model_bundle["normalizer"]

    x = normalizer.transform(input_data)
    x_tensor = torch.from_numpy(x.astype(np.float32))

    score = model.predict_score(x_tensor)

    if score < 0.3:
        severity = "normal"
    elif score < 0.7:
        severity = "warning"
    else:
        severity = "critical"

    return {"anomaly_score": round(score, 4), "severity": severity}


def output_fn(prediction: dict, accept: str = "application/json") -> str:
    """Serialize predict_fn's output dict back to JSON."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction)
