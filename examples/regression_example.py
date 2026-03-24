"""Example 3 — Regression: sine wave approximation and Boston Housing.

Demonstrates:
- Regression losses: MSE, MAE, Huber, LogCosh — compared side by side
- Dataset map / normalise / batch utilities for regression data
- Building deep regression network with Sequential
- Learning rate scheduling (cosine warmup)

Run::

    python examples/regression_example.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import numpy as np
import torch

from lipsync.nn import Dataset, Dropout, Linear, Sequential
from lipsync.losses import HuberLoss, LogCoshLoss, MAELoss, MSELoss

# ===========================================================================
# Part A — Sine wave regression
# ===========================================================================

print("=" * 60)
print("Part A: Sine wave regression")
print("=" * 60)

# 1. Generate data
N = 2000
X_np = np.linspace(-math.pi * 2, math.pi * 2, N, dtype=np.float32).reshape(-1, 1)
y_np = np.sin(X_np).reshape(-1, 1) + np.random.default_rng(0).normal(0, 0.05, (N, 1)).astype(np.float32)

# 2. Dataset utilities
ds = (Dataset
      .from_numpy(X_np, y_np)
      .normalise()
      .shuffle(seed=1))

train_ds, test_ds = ds.split(0.85)
print(f"Train: {train_ds}  Test: {test_ds}")

# 3. Build model
def build_sine_model():
    return Sequential([
        Linear(1, 64,  activation="swish"),
        Linear(64, 128, activation="swish"),
        Linear(128, 64, activation="swish"),
        Linear(64, 1),          # single output
    ])

# 4. Compare regression losses
EPOCHS = 30
RESULTS_SINE = {}

for loss_name in ["mse", "mae", "huber", "log_cosh"]:
    model = build_sine_model()
    model.compile(
        optimizer="adamw",
        loss=loss_name,
        lr=1e-3,
        weight_decay=1e-5,
        scheduler="cosine",
        scheduler_kwargs={"T_max": EPOCHS},
    )
    model.fit(
        train_ds.X, train_ds.y,
        epochs=EPOCHS,
        batch_size=64,
        verbose=False,
    )
    test_m = model.evaluate(test_ds.X, test_ds.y)
    RESULTS_SINE[loss_name] = test_m["loss"]
    print(f"  {loss_name:10s}  test_loss={test_m['loss']:.6f}")

best_sine = min(RESULTS_SINE, key=RESULTS_SINE.get)
print(f"\nBest regression loss: {best_sine} (test={RESULTS_SINE[best_sine]:.6f})")

# 5. Dataset.batch() manual loop demo
print("\nManual batch prediction on 3 batches:")
best_sine_model = build_sine_model()
best_sine_model.compile(optimizer="adamw", loss=best_sine, lr=1e-3)
best_sine_model.fit(train_ds.X, train_ds.y, epochs=EPOCHS, batch_size=64, verbose=False)

for i, (xb, yb) in enumerate(test_ds.batch(64)):
    preds = best_sine_model.predict(xb)
    rmse = ((preds - yb) ** 2).mean().sqrt().item()
    print(f"  Batch {i}: n={len(xb)}  RMSE={rmse:.4f}")
    if i >= 2:
        break


# ===========================================================================
# Part B — Boston Housing regression (sklearn or synthetic fallback)
# ===========================================================================

print("\n" + "=" * 60)
print("Part B: Boston Housing (regression on tabular data)")
print("=" * 60)

def load_housing():
    try:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        return data.data.astype(np.float32), data.target.astype(np.float32)
    except Exception:
        print("sklearn not available — using synthetic housing data (500 samples)")
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 8)).astype(np.float32)
        y = (3.0 + X @ rng.standard_normal(8) + rng.normal(0, 0.3, 500)).astype(np.float32)
        return X, y

X_h, y_h = load_housing()
print(f"Housing: X={X_h.shape}  y range=[{y_h.min():.1f}, {y_h.max():.1f}]")

# Reshape y to (N, 1) for regression output
y_h = y_h.reshape(-1, 1)

ds_h = (Dataset
        .from_numpy(X_h, y_h)
        .normalise()          # normalise both X and y
        .shuffle(seed=7))

train_h, test_h = ds_h.split(0.8)
train_h, val_h  = train_h.split(0.875)
print(f"Train: {train_h}  Val: {val_h}  Test: {test_h}")

# Build deeper network for tabular regression
n_feat = X_h.shape[1]
model_h = Sequential([
    Linear(n_feat, 128, activation="gelu"),
    Dropout(0.2),
    Linear(128, 64, activation="gelu"),
    Dropout(0.1),
    Linear(64, 32,  activation="gelu"),
    Linear(32, 1),
])

model_h.compile(
    optimizer="adamw",
    loss="huber",
    lr=5e-4,
    weight_decay=1e-4,
    scheduler="cosine",
    scheduler_kwargs={"T_max": 50},
)

model_h.fit(
    train_h.X, train_h.y,
    epochs=50,
    batch_size=64,
    val_data=(val_h.X, val_h.y),
    verbose=True,
)

test_m_h = model_h.evaluate(test_h.X, test_h.y)
print(f"\nHousing test Huber loss = {test_m_h['loss']:.4f}")

# Show a few predictions vs actuals
preds_h = model_h.predict(test_h.X[:10])
print("\nSample predictions vs targets (first 10, normalised scale):")
for i in range(10):
    p = preds_h[i, 0].item()
    t = test_h.y[i, 0].item()
    print(f"  pred={p:7.3f}  true={t:7.3f}  err={abs(p-t):.3f}")

# ===========================================================================
# Summary: all regression losses on sine task
# ===========================================================================

print("\n=== Regression loss comparison (sine wave) ===")
print(f"  {'Loss':12s}  {'Test loss':>12s}")
for name, val in sorted(RESULTS_SINE.items(), key=lambda kv: kv[1]):
    print(f"  {name:12s}  {val:12.6f}")
