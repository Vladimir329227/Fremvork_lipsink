"""Example 1 — MNIST digit classification.

Demonstrates:
- Building a network by listing layers (Sequential + Linear + Conv2d)
- Dataset utilities: map, shuffle, split, batch
- 3 different optimisers compared side by side
- Training in a few lines with .compile() / .fit()
- Evaluation with accuracy metric

Run::

    python examples/mnist_example.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

from lipsync.nn import Conv2d, Dataset, Dropout, Flatten, Linear, MaxPool2d, Sequential
from lipsync.optimizers import OPTIMIZER_REGISTRY

# ---------------------------------------------------------------------------
# 1. Load / generate data (downloads MNIST if torchvision available, else random)
# ---------------------------------------------------------------------------

def load_mnist():
    try:
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
        train_raw = MNIST("data/mnist", train=True,  download=True, transform=ToTensor())
        test_raw  = MNIST("data/mnist", train=False, download=True, transform=ToTensor())
        X_train = train_raw.data.float().unsqueeze(1) / 255.0   # (60000, 1, 28, 28)
        y_train = train_raw.targets                              # (60000,)
        X_test  = test_raw.data.float().unsqueeze(1) / 255.0
        y_test  = test_raw.targets
        print("Loaded real MNIST (60 000 train, 10 000 test)")
        return X_train, y_train, X_test, y_test
    except Exception:
        print("torchvision not available — using random data (1 000 samples, 10 classes)")
        X_train = torch.randn(1000, 1, 28, 28)
        y_train = torch.randint(0, 10, (1000,))
        X_test  = torch.randn(200,  1, 28, 28)
        y_test  = torch.randint(0, 10, (200,))
        return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = load_mnist()

# ---------------------------------------------------------------------------
# 2. Dataset utilities
# ---------------------------------------------------------------------------

train_ds = Dataset(X_train, y_train)
test_ds  = Dataset(X_test,  y_test)

# shuffle + split 90/10
train_ds, val_ds = train_ds.shuffle(seed=0).split(0.9)
print(f"Train: {train_ds}  Val: {val_ds}  Test: {test_ds}")

# ---------------------------------------------------------------------------
# 3. Build model: Conv → Pool → Flatten → FC → FC
# ---------------------------------------------------------------------------

def build_cnn():
    return Sequential([
        Conv2d(1, 16, kernel_size=3, padding=1, activation="relu"),
        MaxPool2d(2),
        Conv2d(16, 32, kernel_size=3, padding=1, activation="relu"),
        MaxPool2d(2),
        Flatten(),
        Linear(32 * 7 * 7, 128, activation="relu"),
        Dropout(0.3),
        Linear(128, 10),          # logits for 10 digits
    ])

# ---------------------------------------------------------------------------
# 4. Train with 3 different optimisers and compare
# ---------------------------------------------------------------------------

EPOCHS = 3
BS = 128
RESULTS = {}

for opt_name in ["momentum_sgd", "clipping_sgd", "adamw"]:
    print(f"\n--- Optimiser: {opt_name} ---")
    model = build_cnn()
    compile_kwargs = {
        "optimizer": opt_name,
        "loss": "cross_entropy",
        "lr": 1e-3 if opt_name == "adamw" else 5e-3,
    }
    if opt_name in {"momentum_sgd", "clipping_sgd"}:
        compile_kwargs["momentum"] = 0.9
    if opt_name == "clipping_sgd":
        compile_kwargs["max_norm"] = 1.0
    model.compile(**compile_kwargs)

    model.fit(
        train_ds.X, train_ds.y,
        epochs=EPOCHS,
        batch_size=BS,
        val_data=(val_ds.X, val_ds.y),
    )

    test_metrics = model.evaluate(test_ds.X, test_ds.y)
    RESULTS[opt_name] = test_metrics
    print(f"Test  loss={test_metrics['loss']:.4f}  acc={test_metrics.get('acc', 0):.4f}")

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------

print("\n=== Comparison ===")
for name, m in RESULTS.items():
    print(f"  {name:20s}  acc={m.get('acc', 0):.4f}  loss={m['loss']:.4f}")

# ---------------------------------------------------------------------------
# 6. Batch-iteration example using Dataset.batch()
# ---------------------------------------------------------------------------

print("\n=== Manual batch loop (first 3 batches) ===")
for i, (xb, yb) in enumerate(test_ds.batch(64)):
    pred = model.predict(xb)
    acc = (pred.argmax(1) == yb).float().mean()
    print(f"  Batch {i}: x={tuple(xb.shape)}  batch_acc={acc:.3f}")
    if i >= 2:
        break
