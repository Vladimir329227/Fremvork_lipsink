# Справочник CLI (`lipsync`)

Точка входа: **`cli.py`** или консольная команда **`lipsync`** (см. `[project.scripts]` в `pyproject.toml`).

Общий вид:

```bash
python cli.py <команда> [опции]
lipsync <команда> [опции]
```

---

## Обучение и инференс (фреймворк)

| Команда | Назначение |
|---------|------------|
| `train` | Обучение по `--data-root` и опционально `--config configs/base.yaml` |
| `infer` | Липсинк: `--checkpoint`, `--video`, `--audio`, `--output` |
| `eval` | Метрики (PSNR/SSIM и др.) на сплите датасета |
| `export` | Экспорт генератора в ONNX |
| `serve` | API-сервер (нужен extras `api`) |

Полезные флаги `train`: `--profile cpu-safe|gpu-fast|gpu-quality`, `--early-stopping N`, `--wandb`.

`infer`: `--infer-blend`, `--mouth-composite-mode blend|paste|hard_lower`, `--use-sr`.

---

## Данные

| Команда | Назначение |
|---------|------------|
| `data-download` | Скачивание GRID через Kaggle API → `--output-dir` |
| `data-prepare` | Сырой GRID → формат фреймворка (`--input-root`, `--output-root`) |
| `data-validate` | Проверка целостности датасета |
| `dataset-verify` | Реконструкция на holdout (статик + mel), опционально mux звука |

`data-prepare`: для снижения утечки идентичности между train/val — `--split-by-speaker`.

---

## Пакетный и потоковый режимы

| Команда | Назначение |
|---------|------------|
| `apply-batch` | Одна пара video+audio или список пар из `--pairs-json` |
| `apply-realtime` | Сервер реального времени (нужны зависимости realtime/api по сценарию) |

---

## Wav2Lip (внешний код в `external/Wav2Lip`)

| Команда | Назначение |
|---------|------------|
| `wav2lip-infer` | Инференс официального скрипта: `--face`, `--audio`, `--output` |
| `wav2lip-user-train` | Препроцесс → короткий finetune → итоговый MP4 |

---

## Вспомогательные

| Команда | Назначение |
|---------|------------|
| `static-video` | MP4 из одного изображения + WAV (`--image`, `--audio`, `--output`) |
| `doctor` | Диагностика окружения |
| `benchmark` | Синтетический прогон производительности |
| `profile-realtime` | Агрегация `*_meta.json` → p50/p95/p99 |

---

Полный список аргументов для каждой команды:

```bash
lipsync <команда> --help
```
