"""
Train the BiLSTM anomaly detector end to end.

Pipeline:
    1. Generate synthetic sensor data
    2. Slice into windows, stratified train/val/test split
    3. Fit normaliser on TRAIN ONLY (avoids data leakage)
    4. Train with weighted BCE loss (anomalies are ~5% of data — without
       weighting, the model could get 95% accuracy by predicting "normal"
       for everything, which is useless)
    5. Track best model by validation F1 (not accuracy — see above)
    6. Evaluate on held-out test set
    7. Save model weights + normaliser stats together, so inference
       never has a "which normaliser did this model use" ambiguity

Usage:
    cd asset_health_monitor
    python models/train.py
    python models/train.py --epochs 30 --lr 0.001
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate_sensor_data import generate
from models.dataset import build_windows, train_val_test_split, Normalizer, SensorWindowDataset
from models.bilstm import BiLSTMAnomalyDetector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model: nn.Module, loader: DataLoader, threshold: float = 0.5) -> dict:
    """Compute precision/recall/F1/accuracy on a data loader."""
    model.eval()
    tp = fp = tn = fn = 0
    losses = []
    criterion = nn.BCELoss()

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X)
            losses.append(criterion(preds, y).item())

            pred_labels = (preds >= threshold).float()
            tp += ((pred_labels == 1) & (y == 1)).sum().item()
            fp += ((pred_labels == 1) & (y == 0)).sum().item()
            tn += ((pred_labels == 0) & (y == 0)).sum().item()
            fn += ((pred_labels == 0) & (y == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

    return {
        "loss": float(np.mean(losses)),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def train(args):
    print(f"Device: {DEVICE}\n")

    # --- 1. Data -----------------------------------------------------------
    print(f"Generating {args.days} days of synthetic sensor data...")
    df = generate(days=args.days)

    print(f"Building windows (size={args.window_size}, stride={args.stride})...")
    X, y, mids = build_windows(df, window_size=args.window_size, stride=args.stride)
    print(f"  Total windows: {len(y)}, anomaly rate: {y.mean()*100:.2f}%")

    splits = train_val_test_split(X, y, mids, seed=args.seed)
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]
    print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    # --- 2. Normalisation (fit on TRAIN only) -------------------------------
    import pandas as pd
    train_df = pd.DataFrame(X_train.reshape(-1, X_train.shape[-1]),
                             columns=["vibration_rms", "temperature_c", "pressure_bar", "current_amp"])
    normalizer = Normalizer.fit(train_df)

    train_ds = SensorWindowDataset(X_train, y_train, normalizer)
    val_ds   = SensorWindowDataset(X_val,   y_val,   normalizer)
    test_ds  = SensorWindowDataset(X_test,  y_test,  normalizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # --- 3. Model, loss, optimiser ------------------------------------------
    model = BiLSTMAnomalyDetector(
        n_sensors=4, hidden_size=args.hidden_size, num_layers=args.num_layers
    ).to(DEVICE)

    # Weighted BCE — upweight the rare anomaly class so the model can't just
    # predict "normal" for everything and call it a day.
    pos_weight_ratio = (1 - y_train.mean()) / y_train.mean()
    print(f"  Positive class upweighted {pos_weight_ratio:.1f}x in loss\n")

    def weighted_bce(preds, targets):
        weights = torch.where(targets == 1, pos_weight_ratio, 1.0)
        eps = 1e-7
        preds = preds.clamp(eps, 1 - eps)
        loss = -(targets * torch.log(preds) + (1 - targets) * torch.log(1 - preds))
        return (loss * weights).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- 4. Training loop ----------------------------------------------------
    best_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = weighted_bce(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_metrics = evaluate(model, val_loader)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), **val_metrics})

        marker = ""
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = "  ← best so far"

        print(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"train_loss={np.mean(train_losses):.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_precision={val_metrics['precision']:.3f} | "
            f"val_recall={val_metrics['recall']:.3f} | "
            f"val_f1={val_metrics['f1']:.3f}{marker}"
        )

    # --- 5. Restore best model, evaluate on test set --------------------------
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader)

    print(f"\n{'='*60}")
    print("Best model (by validation F1) — held-out TEST set results:")
    print(f"{'='*60}")
    print(f"  Precision : {test_metrics['precision']}")
    print(f"  Recall    : {test_metrics['recall']}")
    print(f"  F1        : {test_metrics['f1']}")
    print(f"  Accuracy  : {test_metrics['accuracy']}")
    print(f"  Confusion : TP={test_metrics['tp']} FP={test_metrics['fp']} "
          f"TN={test_metrics['tn']} FN={test_metrics['fn']}")

    # --- 6. Save model + normaliser + metadata together -----------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, out_dir / "bilstm_weights.pt")

    metadata = {
        "model_config": {
            "n_sensors": 4,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "window_size": args.window_size,
        },
        "normalizer": normalizer.to_dict(),
        "sensor_cols": ["vibration_rms", "temperature_c", "pressure_bar", "current_amp"],
        "test_metrics": test_metrics,
        "training_history": history,
    }
    with open(out_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model weights  → {out_dir / 'bilstm_weights.pt'}")
    print(f"Saved metadata       → {out_dir / 'model_metadata.json'}")
    print("\nPhase 2 complete! Next step → Phase 3: build the MCP servers.")


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM anomaly detector")
    parser.add_argument("--days",        type=int,   default=30)
    parser.add_argument("--window-size", type=int,   default=50)
    parser.add_argument("--stride",      type=int,   default=5)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int,   default=64)
    parser.add_argument("--num-layers",  type=int,   default=2)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--output-dir",  type=str,   default="models/saved")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
