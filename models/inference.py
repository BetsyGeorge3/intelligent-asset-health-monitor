"""
Inference wrapper for the trained BiLSTM anomaly detector.

This is the ONLY file the agent (Phase 4) and the sensor MCP server
(Phase 3) should import from `models/`. It hides PyTorch, normalisation,
and checkpoint-loading details behind one clean function:

    score = get_anomaly_score(machine_id, last_n=50)

Loads the model once (lazy singleton) so repeated calls during an agent
run don't reload weights from disk every time.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bilstm import BiLSTMAnomalyDetector
from models.dataset import Normalizer, SENSOR_COLS
from data.reader import load_window, SensorWindow

MODEL_DIR = Path(__file__).parent / "saved"


@dataclass
class AnomalyResult:
    """Structured result returned to the agent — everything it needs to reason."""
    machine_id:    str
    score:         float            # 0.0–1.0 anomaly probability
    severity:      str              # "normal" | "warning" | "critical"
    sensor_flags:  list[str]        # sensors that look elevated vs. their own history
    latest_values: dict[str, float]
    window:        SensorWindow     # full window, in case the agent wants more detail

    def to_dict(self) -> dict:
        """JSON-serialisable view — used when this crosses into an MCP response."""
        return {
            "machine_id": self.machine_id,
            "anomaly_score": round(self.score, 4),
            "severity": self.severity,
            "sensor_flags": self.sensor_flags,
            "latest_values": {k: round(v, 4) for k, v in self.latest_values.items()},
        }


class _ModelSingleton:
    """Lazily loads the model + normaliser once per process."""
    _model: BiLSTMAnomalyDetector | None = None
    _normalizer: Normalizer | None = None
    _window_size: int = 50

    @classmethod
    def get(cls) -> tuple[BiLSTMAnomalyDetector, Normalizer, int]:
        if cls._model is None:
            weights_path  = MODEL_DIR / "bilstm_weights.pt"
            metadata_path = MODEL_DIR / "model_metadata.json"

            if not weights_path.exists():
                raise FileNotFoundError(
                    f"No trained model found at {weights_path}. "
                    f"Run `python models/train.py` first."
                )

            with open(metadata_path) as f:
                metadata = json.load(f)

            cfg = metadata["model_config"]
            model = BiLSTMAnomalyDetector(
                n_sensors=cfg["n_sensors"],
                hidden_size=cfg["hidden_size"],
                num_layers=cfg["num_layers"],
            )
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            model.eval()

            cls._model = model
            cls._normalizer = Normalizer.from_dict(metadata["normalizer"])
            cls._window_size = cfg["window_size"]

        return cls._model, cls._normalizer, cls._window_size


def _severity_from_score(score: float) -> str:
    """Maps a continuous anomaly score to a discrete severity bucket."""
    if score < 0.3:
        return "normal"
    elif score < 0.7:
        return "warning"
    else:
        return "critical"


def _flag_elevated_sensors(window: SensorWindow, z_threshold: float = 2.0) -> list[str]:
    """
    Identify which sensors look unusual within their OWN recent window
    (z-score of latest reading vs. the window's own mean/std).

    This is separate from the model score — it gives the agent a
    human-readable reason ("vibration_rms is elevated") rather than just
    a single opaque number.
    """
    flags = []
    stats  = window.summary_stats()
    latest = window.latest_values()

    for sensor in SENSOR_COLS:
        mean, std = stats[sensor]["mean"], stats[sensor]["std"]
        if std < 1e-6:
            continue
        z = (latest[sensor] - mean) / std
        if abs(z) >= z_threshold:
            flags.append(sensor)

    return flags


def get_anomaly_score(machine_id: str, last_n: int | None = None) -> AnomalyResult:
    """
    Run the trained BiLSTM on a machine's most recent sensor window.

    Args:
        machine_id: e.g. "PUMP-01"
        last_n: window size — defaults to the size the model was trained on.
                Only override this if you know what you're doing; the model
                expects exactly the window size it was trained with.

    Returns:
        AnomalyResult with score, severity bucket, and flagged sensors.
    """
    model, normalizer, trained_window_size = _ModelSingleton.get()
    window_size = last_n or trained_window_size

    window = load_window(machine_id, last_n=window_size)

    x = normalizer.transform(window.as_array)          # (window_size, n_sensors)
    x_tensor = torch.from_numpy(x.astype(np.float32))

    score = model.predict_score(x_tensor)
    severity = _severity_from_score(score)
    flags = _flag_elevated_sensors(window)

    return AnomalyResult(
        machine_id=machine_id,
        score=score,
        severity=severity,
        sensor_flags=flags,
        latest_values=window.latest_values(),
        window=window,
    )


if __name__ == "__main__":
    from data.store import get_all_machine_ids

    for mid in get_all_machine_ids():
        result = get_anomaly_score(mid)
        print(f"\n{mid}")
        print(f"  Score        : {result.score:.4f}")
        print(f"  Severity     : {result.severity}")
        print(f"  Flagged      : {result.sensor_flags or 'none'}")
        print(f"  Latest values: {result.latest_values}")
