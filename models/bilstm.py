"""
BiLSTM anomaly detection model.

Architecture:
    Input  (batch, seq_len, n_sensors)
      → Bidirectional LSTM (captures patterns in both time directions)
      → Attention-free pooling (last timestep from both directions)
      → Fully-connected head
      → Sigmoid output: anomaly score in [0, 1]

This is a binary sequence classifier: given a window of sensor readings,
predict the probability that the window represents an anomalous /
fault-developing state rather than normal operation.

Why BiLSTM and not a plain LSTM:
    A developing fault (e.g. bearing wear) shows a *ramp* — gradually
    rising vibration. Reading the window backwards as well as forwards
    helps the model recognise both "rising toward a fault" and
    "recovering from one" as distinct patterns, which a one-directional
    LSTM tends to blur together.
"""

import torch
import torch.nn as nn


class BiLSTMAnomalyDetector(nn.Module):
    """
    Bidirectional LSTM binary classifier for sensor anomaly detection.

    Args:
        n_sensors:   number of input sensor channels (default 4 —
                     vibration_rms, temperature_c, pressure_bar, current_amp)
        hidden_size: LSTM hidden state size per direction
        num_layers:  number of stacked LSTM layers
        dropout:     dropout applied between LSTM layers and in the FC head
    """

    def __init__(
        self,
        n_sensors: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.n_sensors   = n_sensors
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Bidirectional LSTM: output is 2*hidden_size (forward + backward concatenated)
        self.lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification head: 2*hidden_size → hidden_size → 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_sensors) float tensor

        Returns:
            (batch,) tensor of anomaly probabilities in [0, 1]
        """
        # lstm_out: (batch, seq_len, 2*hidden_size)
        # h_n:      (2*num_layers, batch, hidden_size) — final hidden state per direction/layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the final layer's forward and backward hidden states and concatenate.
        # h_n is ordered [layer0_fwd, layer0_bwd, layer1_fwd, layer1_bwd, ...]
        # so the last layer's forward/backward states are the last two entries.
        h_fwd = h_n[-2]   # (batch, hidden_size) — last layer, forward direction
        h_bwd = h_n[-1]   # (batch, hidden_size) — last layer, backward direction
        final = torch.cat([h_fwd, h_bwd], dim=1)   # (batch, 2*hidden_size)

        logits = self.head(final).squeeze(-1)      # (batch,)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_score(self, x: torch.Tensor) -> float:
        """
        Convenience method for single-sample inference.

        Args:
            x: (seq_len, n_sensors) — a single window, no batch dimension

        Returns:
            Anomaly score as a plain Python float.
        """
        self.eval()
        if x.dim() == 2:
            x = x.unsqueeze(0)   # add batch dimension → (1, seq_len, n_sensors)
        score = self.forward(x)
        return float(score.item())


if __name__ == "__main__":
    # Quick architecture smoke-test with random data
    model = BiLSTMAnomalyDetector(n_sensors=4, hidden_size=64, num_layers=2)
    dummy = torch.randn(8, 50, 4)   # batch=8, seq_len=50, sensors=4
    out = model(dummy)
    print(f"Output shape: {out.shape}")     # expect (8,)
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"Param count : {sum(p.numel() for p in model.parameters()):,}")
