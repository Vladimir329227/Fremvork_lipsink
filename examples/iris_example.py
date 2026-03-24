"""Example 2 — Iris flower classification.

Demonstrates:
- Dataset.from_numpy(), normalise(), shuffle(), split()
- Multi-class classification with CrossEntropy and LabelSmoothing losses
- Building a small fully-connected network with Sequential + Linear
- Comparing SGD, MomentumSGD, AdamW optimisers
- predict_classes() for inference

Run::

    python examples/iris_example.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from lipsync.nn import Dataset, Linear, Sequential

# ---------------------------------------------------------------------------
# 1. Load Iris dataset (sklearn or hardcoded fallback)
# ---------------------------------------------------------------------------

def load_iris():
    try:
        from sklearn.datasets import load_iris as _load
        data = _load()
        return data.data.astype(np.float32), data.target.astype(np.int64)
    except ImportError:
        print("sklearn not found — using built-in Iris data (150 samples)")
        # Subset of Iris: 150 samples, 4 features, 3 classes
        rng = np.random.default_rng(0)
        X = np.vstack([
            rng.normal([5.0, 3.4, 1.5, 0.2], 0.4, (50, 4)),  # setosa
            rng.normal([5.9, 2.8, 4.5, 1.3], 0.5, (50, 4)),  # versicolor
            rng.normal([6.6, 3.0, 5.6, 2.0], 0.5, (50, 4)),  # virginica
        ]).astype(np.float32)
        y = np.repeat([0, 1, 2], 50).astype(np.int64)
        return X, y


X_np, y_np = load_iris()
print(f"Iris: X={X_np.shape}  classes={np.unique(y_np)}")

# ---------------------------------------------------------------------------
# 2. Dataset utilities
# ---------------------------------------------------------------------------

ds = (Dataset
      .from_numpy(X_np, y_np)
      .normalise()           # z-score features
      .shuffle(seed=42))

train_ds, test_ds = ds.split(0.8)
train_ds, val_ds  = train_ds.split(0.875)   # 70 / 10 / 20 split

print(f"Train: {train_ds}  Val: {val_ds}  Test: {test_ds}")

# ---------------------------------------------------------------------------
# 3. Build model: 4 → 64 → 32 → 3
# ---------------------------------------------------------------------------

def build_iris_model():
    return Sequential([
        Linear(4,  64, activation="relu"),
        Linear(64, 32, activation="relu"),
        Linear(32,  3),        # 3-class logits
    ])

# ---------------------------------------------------------------------------
# 4. Train with 3 optimisers, 2 loss functions
# ---------------------------------------------------------------------------

configs = [
    # (optimizer_name, loss_name, lr, extra_opt_kwargs, extra_loss_kwargs)
    ("sgd",          "cross_entropy",   5e-2, {},                          {}),
    ("momentum_sgd", "cross_entropy",   1e-2, {"momentum": 0.9},           {}),
    ("adamw",        "label_smoothing", 5e-3, {},                          {"num_classes": 3, "smoothing": 0.1}),
]

EPOCHS = 50
RESULTS = {}

for opt_name, loss_name, lr, extra_opt, extra_loss in configs:
    label = f"{opt_name}/{loss_name}"
    print(f"\n--- {label} ---")

    model = build_iris_model()
    model.compile(optimizer=opt_name, loss=loss_name, lr=lr,
                  loss_kwargs=extra_loss or None, **extra_opt)

    model.fit(
        train_ds.X, train_ds.y,
        epochs=EPOCHS,
        batch_size=32,
        val_data=(val_ds.X, val_ds.y),
        verbose=False,
    )

    test_m = model.evaluate(test_ds.X, test_ds.y)
    RESULTS[label] = test_m
    print(f"  Test  loss={test_m['loss']:.4f}  acc={test_m.get('acc', 0):.4f}")

# ---------------------------------------------------------------------------
# 5. Best model — predict classes
# ---------------------------------------------------------------------------

best_label = max(RESULTS, key=lambda k: RESULTS[k].get("acc", 0))
print(f"\nBest config: {best_label}  acc={RESULTS[best_label].get('acc', 0):.4f}")

# Re-train best model for demo
best_opt, best_loss, best_lr, best_extra_opt, best_extra_loss = next(
    (o, l, lr, eo, el) for o, l, lr, eo, el in configs if f"{o}/{l}" == best_label
)
best_model = build_iris_model()
best_model.compile(optimizer=best_opt, loss=best_loss, lr=best_lr,
                   loss_kwargs=best_extra_loss or None, **best_extra_opt)
best_model.fit(train_ds.X, train_ds.y, epochs=EPOCHS, batch_size=32, verbose=False)

preds = best_model.predict_classes(test_ds.X)
labels = test_ds.y
class_names = ["setosa", "versicolor", "virginica"]
print("\nSample predictions (first 10):")
for i in range(min(10, len(preds))):
    p, t = preds[i].item(), labels[i].item()
    print(f"  pred={class_names[p]:12s}  true={class_names[t]:12s}  {'✓' if p == t else '✗'}")

# ---------------------------------------------------------------------------
# 6. Dataset helper demo
# ---------------------------------------------------------------------------

print("\n=== Dataset utilities demo ===")
print(f"  k-fold (k=5) fold sizes:")
for fold_i, (tr, va) in enumerate(ds.k_fold(k=5)):
    print(f"    Fold {fold_i}: train={len(tr)}  val={len(va)}")
