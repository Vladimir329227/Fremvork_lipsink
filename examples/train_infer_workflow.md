# Пример: от подготовки GRID до инференса

## Шаг 0. Окружение

```bash
pip install -e .
lipsync doctor
```

Убедитесь, что **ffmpeg** доступен в терминале, если нужен звук в итоговом MP4.

## Шаг 1. Данные GRID

Скачивание (нужен настроенный Kaggle API, см. `kaggle.json`):

```bash
lipsync data-download --output-dir ./data/grid_raw
```

Подготовка в формат фреймворка (пути к распакованному GRID):

```bash
lipsync data-prepare --input-root ./data/grid_raw --output-root ./data/grid_prepared --val-ratio 0.1 --split-by-speaker
```

Проверка:

```bash
lipsync data-validate --data-root ./data/grid_prepared
```

## Шаг 2. Обучение

```bash
lipsync train --config configs/base.yaml --data-root ./data/grid_prepared --epochs 50 --batch-size 8 --device cuda
```

Чекпоинты по умолчанию окажутся в каталоге из конфига (часто `checkpoints/`). Имя лучшего файла смотрите в логах или в файловой системе.

## Шаг 3. Оценка и sanity-check

```bash
lipsync eval --checkpoint checkpoints/best.pt --data-root ./data/grid_prepared --split val
```

Реконструкция на нескольких клипах (видео в `--out-dir`):

```bash
lipsync dataset-verify --checkpoint checkpoints/best.pt --data-root ./data/grid_prepared --split val --num-clips 5 --out-dir ./verify_out
```

## Шаг 4. Инференс на своём ролике

```bash
lipsync infer --checkpoint checkpoints/best.pt --video my_face.mp4 --audio narration.wav --output result.mp4
```

Если в ROI виден «прямоугольник» от генератора, попробуйте явный бленд:

```bash
lipsync infer --checkpoint checkpoints/best.pt --video my_face.mp4 --audio narration.wav --output result.mp4 --mouth-composite-mode blend
```
