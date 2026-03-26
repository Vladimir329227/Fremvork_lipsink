# Changelog

## 0.2.0 - Framework-grade upgrade

### Added
- Runtime compatibility layer with fail-fast diagnostics:
  - `lipsync/runtime/compat.py`
  - Dependency checks for `torch`, `torchvision`, `torchaudio`, `cv2`, `onnxruntime`
- Runtime execution profiles:
  - `cpu-safe`
  - `gpu-fast`
  - `gpu-quality`
- Deterministic training mode:
  - unified seed setup for Python/NumPy/PyTorch
  - deterministic backend flags

### Checkpointing
- Versioned checkpoint schema v2:
  - `meta.schema_version`
  - `meta.created_at`
  - `meta.git_hash`
  - `meta.config_hash`
  - `meta.training_signature`
- Automatic migration from legacy checkpoint format to v2
- Checkpoint validation before loading

### Config Validation
- Strict config schema validation with fail-fast errors:
  - `lipsync/config/schema.py`
- Validation integrated into high-level trainer initialization

### Data Validation
- Dataset integrity suite:
  - metadata checks
  - required file/path checks
  - frame/lips count consistency
  - mel tensor and landmarks shape checks
  - suspicious FPS and short audio warnings
- New CLI command:
  - `lipsync data-validate --data-root ... --split ...`

### Operations / Diagnostics
- New operational tools module:
  - `lipsync/ops.py`
- New CLI commands:
  - `lipsync doctor`
  - `lipsync benchmark`
  - `lipsync profile-realtime`

### Realtime & API Hardening
- Realtime pipeline improvements:
  - graceful degradation when detector/SR fails
  - latency tracking with p50/p95/p99
  - estimated FPS and realtime factor reporting
- API robustness:
  - request size limits
  - bounded websocket queue/concurrency

### Plugin System
- Plugin interfaces for:
  - models
  - losses
  - optimizers
  - preprocessors
- Plugin registry with dynamic discovery via Python entry points

### Packaging
- Added `pyproject.toml`
- Added optional extras:
  - `.[api]`
  - `.[realtime]`
  - `.[sr]`
- Console script entrypoint:
  - `lipsync`

### Quality Gates
- Added test suite:
  - unit tests (factories)
  - integration smoke tests
  - regression guards (shape/NaN invariants)
- Added CI workflow (`.github/workflows/ci.yml`) running `pytest`

### Documentation
- Expanded README sections:
  - Runtime Profiles
  - Framework Operations
  - Checkpoint Schema v2
  - Plugin API
  - Reproducibility
  - Troubleshooting

## 0.1.0 - Initial release

- Core lip-sync architecture (audio encoder, identity encoder, generator, discriminator)
- Training loop and high-level API
- Realtime and batch inference pipelines
- REST + WebSocket API
- Activation/loss/optimizer registries
- Classic ML examples (MNIST/Iris/regression)
