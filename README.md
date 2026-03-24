# LipSync Framework

A PyTorch framework for training audio-driven lip-synchronisation neural networks.
Supports real-time inference, head rotation handling, optional super-resolution,
and a high-level API for training in a few lines of code.

---

## Features

| Feature | Details |
|---|---|
| **Sequential model builder** | `Sequential([Linear(784,256,'relu'), Dropout(0.3), Linear(256,10)])` |
| **Dataset utilities** | `Dataset` — map, map_x, filter, shuffle, split, batch, k_fold, normalise, one_hot |
| **Optimisers (≥3)** | SGD, Momentum SGD (+ Nesterov), Gradient-Clipping SGD, AdamW, Lion, Lookahead |
| **Activations** | ReLU, LeakyReLU, GELU, Swish/SiLU, Mish, Sigmoid, Tanh, Softmax (+ 4 more) |
| **Losses** | MSE, MAE, Huber, LogCosh (regression) · BCE, CrossEntropy, Focal, LabelSmoothing (classification) · Perceptual, Sync, Temporal, Identity, Adversarial (lip-sync) |
| **Train in few lines** | `.compile(optimizer, loss, lr)` → `.fit(X, y, epochs)` |
| **Classic examples** | MNIST classification, Iris classification, sine regression + California Housing |
| **Lip-sync architecture** | Conformer audio encoder + ResNet identity encoder + U-Net generator with cross-attention |
| **Head rotation** | 3DMM pose estimation — expression parameters replaced, pose & shape preserved |
| **Real-time** | < 100 ms latency, ONNX export, WebSocket streaming API |
| **Super-resolution** | GFPGAN v1.4, CodeFormer, Real-ESRGAN (optional) |
| **Evaluation** | PSNR, SSIM, LMD, Lip-LMD, SyncScore |
| **REST + WebSocket API** | FastAPI for easy integration into products |

---

## Installation

```bash
git clone <repo>
cd lipsync_framework

pip install -r requirements.txt

# Optional: super-resolution backends
pip install gfpgan            # GFPGAN
pip install basicsr           # CodeFormer (requires basicsr)
pip install realesrgan        # Real-ESRGAN
```

---

## Quick Start

### Build a neural network by listing layers

```python
from lipsync import Sequential, Linear, Conv2d, Flatten, Dropout, Dataset

# Fully-connected classifier
model = Sequential([
    Linear(784, 256, activation="relu"),
    Dropout(0.3),
    Linear(256, 128, activation="relu"),
    Linear(128, 10),
])

model.compile(optimizer="adamw", loss="cross_entropy", lr=1e-3)
model.fit(X_train, y_train, epochs=20, batch_size=64, val_data=(X_val, y_val))
print(model.evaluate(X_test, y_test))
preds = model.predict_classes(X_test)
```

### Dataset utilities

```python
from lipsync import Dataset

ds = Dataset.from_numpy(X_np, y_np)
ds = ds.normalise().shuffle(seed=42)
train_ds, val_ds = ds.split(0.8)

# Map transform
ds = ds.map_x(lambda x: x / 255.0)

# Mini-batch iteration
for xb, yb in train_ds.batch(32, shuffle=True):
    ...

# k-fold cross-validation
for train_fold, val_fold in ds.k_fold(k=5):
    ...
```

### Training in 3 lines (lip-sync)

```python
from lipsync import LipSyncTrainer
from lipsync.data import LipSyncDataset

train_ds = LipSyncDataset("data/", split="train", augment=True)
val_ds   = LipSyncDataset("data/", split="val")

trainer = LipSyncTrainer.from_config("configs/base.yaml")
trainer.fit(train_ds, val_ds)
```

### Training with full control

```python
from lipsync import LipSyncTrainer, LipSyncConfig, EarlyStopping

config = LipSyncConfig(
    model={"gen_base_ch": 128, "gen_depth": 5},
    optimizer={"name": "lion", "lr": 3e-5},
    losses={"w_sync": 0.8, "adv_mode": "hinge"},
    epochs=200,
    batch_size=16,
    fp16=True,
    use_super_resolution=True,
    sr_backend="gfpgan",
    log_wandb=True,
)

trainer = LipSyncTrainer(config)
trainer.fit(train_ds, val_ds, callbacks=[EarlyStopping(patience=20)])
```

### Inference

```python
result = trainer.predict(audio="voice.wav", video="face.mp4")
result.save("output.mp4")
```

### Resume from checkpoint

```python
trainer = LipSyncTrainer.from_checkpoint("checkpoints/best_model.pt")
result = trainer.predict(audio="new_speech.wav", video="face.mp4")
```

---

## CLI

```bash
# Train
python cli.py train --data-root data/ --config configs/base.yaml --epochs 100

# Train with specific optimizer
python cli.py train --data-root data/ --optimizer momentum_sgd --lr 1e-3

# Inference
python cli.py infer --checkpoint checkpoints/best_model.pt \
                    --audio voice.wav --video face.mp4 --output result.mp4

# Apply super-resolution
python cli.py infer --checkpoint checkpoints/best_model.pt \
                    --audio voice.wav --video face.mp4 --use-sr

# Start REST/WebSocket API server
python cli.py serve --checkpoint checkpoints/best_model.pt --port 8000

# Export to ONNX
python cli.py export --checkpoint checkpoints/best_model.pt --output model.onnx

# Evaluate
python cli.py eval --checkpoint checkpoints/best_model.pt --data-root data/ --split test
```

---

## Classic Examples

```bash
python examples/iris_example.py        # Iris 3-class classification
python examples/regression_example.py  # Sine regression + California Housing
python examples/mnist_example.py       # MNIST digit classification (CNN)
```

## Project Structure

```
lipsync_framework/
├── lipsync/                       # Main Python package
│   ├── nn/                        # Sequential builder, Layer helpers, Dataset utilities
│   ├── activations/               # ReLU, GELU, Swish, Mish, Sigmoid, Tanh, Softmax …
│   ├── data/
│   │   ├── datasets/              # LipSyncDataset, VideoDataset
│   │   ├── preprocessing/         # AudioPreprocessor, VideoPreprocessor
│   │   └── augmentation/          # SpecAugment, VideoAugmentation
│   ├── models/
│   │   ├── audio_encoder/         # Conformer (mel → audio embedding)
│   │   ├── identity_encoder/      # ResNet-18 (face → identity embedding)
│   │   ├── pose_estimator/        # MobileNetV3 → 3DMM params
│   │   ├── generator/             # U-Net with cross-attention
│   │   ├── discriminator/         # PatchGAN + SyncNet
│   │   └── super_resolution/      # GFPGAN / CodeFormer / Real-ESRGAN wrapper
│   ├── losses/                    # Regression, classification, lip-sync losses
│   ├── optimizers/                # SGD, MomentumSGD, ClippingSGD, AdamW, Lion, Lookahead
│   ├── training/                  # Trainer, callbacks
│   ├── inference/
│   │   ├── realtime/              # Ring buffer, < 100ms pipeline, ONNX export
│   │   └── batch/                 # High-quality offline processor
│   ├── evaluation/                # PSNR, SSIM, LMD, SyncScore
│   └── api/                       # FastAPI + WebSocket
├── examples/                      # iris_example.py, regression_example.py, mnist_example.py
├── configs/
│   ├── base.yaml                  # Default config
│   ├── model/lipsync_v2.yaml      # Larger model config
│   ├── optimizer/                 # Per-optimizer presets
│   └── loss/default.yaml
├── cli.py                         # Command-line interface
└── requirements.txt
```

---

## Optimisers

| Name (config key) | Class | Notes |
|---|---|---|
| `sgd` | `SGD` | Vanilla SGD |
| `momentum_sgd` | `MomentumSGD` | SGD + momentum + optional Nesterov |
| `clipping_sgd` | `GradientClippingSGD` | SGD with per-step gradient clipping |
| `adamw` | `AdamW` | Decoupled weight decay (default) |
| `lion` | `Lion` | Sign-based, memory-efficient |
| Any of above | `Lookahead(base)` | Wrap with `lookahead: true` |

## Activation Functions

Build any activation via `build_activation("name")`:
`relu`, `leaky_relu`, `gelu`, `swish`, `silu`, `mish`, `sigmoid`, `tanh`, `softmax`, `elu`, `prelu`, `hardswish`

## Loss Functions

Build any loss via `build_loss("name")`:
- **Regression**: `mse`, `mae` / `l1`, `huber`, `log_cosh`
- **Classification**: `bce`, `cross_entropy`, `focal`, `label_smoothing`
- **Lip-sync**: `perceptual`, `sync`, `temporal`, `identity`, `adversarial`

---

## REST API

Start server:
```bash
python cli.py serve --checkpoint checkpoints/best_model.pt
```

Endpoints:
- `GET  /health` — health check
- `POST /lipsync/batch` — upload video + audio, returns lip-synced MP4
- `POST /lipsync/frame` — single frame processing (base64 JSON)
- `WS   /ws/lipsync` — WebSocket streaming for real-time applications

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **PSNR** | Peak Signal-to-Noise Ratio (image quality) |
| **SSIM** | Structural Similarity Index |
| **LMD** | Landmark Mean Distance (normalised by inter-ocular distance) |
| **Lip-LMD** | LMD restricted to lip landmarks (48–68) |
| **SyncScore** | Audio-visual cosine similarity via SyncNet |

---



## GRID Corpus Preparation

If you use GRID from Kaggle (`s*_processed/*.mpg` + `s*_processed/align/*.align`),
use the ready script:

```bash
python scripts/prepare_grid_dataset.py   --input-root data   --output-root data/processed   --speakers s7_processed,s31_processed   --device cpu
```

Quick test on a small subset:

```bash
python scripts/prepare_grid_dataset.py   --input-root data   --output-root data/processed_debug   --speakers s7_processed   --limit 20   --overwrite
```

Then train:

```bash
python cli.py train --data-root data/processed --config configs/base.yaml
```

## Dataset Format

Pre-process your dataset into the following layout:

```
data/
├── train_metadata.json
├── val_metadata.json
└── samples/
    ├── 00001/
    │   ├── audio.pt         # (T, 80) mel tensor
    │   ├── frames/          # face crops as JPEG
    │   ├── lips/            # lip-region crops as JPEG
    │   └── landmarks.pt     # (N_frames, 68, 2)
    └── 00002/
        └── …
```

`metadata.json` is a list of dicts with keys: `id`, `n_frames`, `fps`, `speaker`.
