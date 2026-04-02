# Быстрый старт

## Требования

- Python **≥ 3.10**
- **PyTorch** с поддержкой вашего GPU (или CPU)
- Для выходных MP4 со звуком — **ffmpeg** в `PATH`

## Установка

Из корня репозитория:

```bash
pip install -e .
```

Опциональные группы зависимостей (из `pyproject.toml`):

```bash
pip install -e ".[api]"       # REST / WebSocket (serve, apply-realtime с HTTP)
pip install -e ".[realtime]" # ONNX, onnxruntime, face-alignment
pip install -e ".[sr]"       # суперразрешение (GFPGAN и др.)
pip install -e ".[all]"      # всё перечисленное
```

Либо классический вариант:

```bash
pip install -r requirements.txt
```

## Проверка окружения

```bash
python cli.py doctor
# или
lipsync doctor
```

Сохранить отчёт в JSON:

```bash
lipsync doctor --output doctor_report.json
```

## Два пути: свой тренер vs Wav2Lip

| Путь | Когда использовать |
|------|---------------------|
| **`lipsync train` / `infer`** | Своя архитектура фреймворка, чекпоинты `LipSyncTrainer` |
| **`lipsync wav2lip-infer`** | Готовая модель Rudrabha/Wav2Lip (`external/Wav2Lip`, веса `.pth` вручную) |
| **`lipsync wav2lip-user-train`** | Короткий дообучающий пайплайн Wav2Lip на вашем видео |

## Минимальный сценарий после подготовки датасета

1. Данные в формате фреймворка (см. `data-prepare` в [CLI](cli-reference.md)).
2. Обучение:

```bash
lipsync train --config configs/base.yaml --data-root path/to/prepared_data
```

3. Инференс:

```bash
lipsync infer --checkpoint path/to/best.pt --video input.mp4 --audio speech.wav --output out.mp4
```

Подробные примеры — в [`examples/`](../examples/README.md).
