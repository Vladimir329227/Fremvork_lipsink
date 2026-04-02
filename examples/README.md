# Примеры использования

Все пути замените на свои. Команды одинаковы для `python cli.py ...` и `lipsync ...` из установленного пакета.

## 1. Обучение и инференс (чекпоинт фреймворка)

См. подробный пошаговый сценарий: [`train_infer_workflow.md`](train_infer_workflow.md).

Кратко:

```bash
lipsync train --config configs/base.yaml --data-root ./data/my_grid_prepared
lipsync infer --checkpoint checkpoints/best.pt --video face.mp4 --audio voice.wav --output lipsync_out.mp4
```

С блендом поверх дефолтов чекпоинта:

```bash
lipsync infer --checkpoint checkpoints/best.pt --video face.mp4 --audio voice.wav --output out.mp4 --infer-blend
```

## 2. Пакетная обработка нескольких пар (JSON)

Файл [`pairs_batch.json`](pairs_batch.json) — шаблон для `apply-batch`:

```bash
lipsync apply-batch --checkpoint checkpoints/best.pt --pairs-json examples/pairs_batch.json
```

## 3. Статичное видео под аудио (без нейросети фреймворка)

Удобно как заглушка или вход для внешних пайплайнов:

```bash
lipsync static-video --image portrait.png --audio speech.wav --output static_for_wav2lip.mp4
```

## 4. Wav2Lip (готовые веса)

Нужны веса `.pth` в `external/Wav2Lip` (см. официальный README Wav2Lip). Затем:

```bash
lipsync wav2lip-infer --face input.mp4 --audio speech.wav --output wav2lip_out.mp4
```

Короткий пользовательский пайплайн (дообучение + инференс):

```bash
lipsync wav2lip-user-train --train-video my_talk.mp4 --audio new_voice.wav --max-steps 500
```

## 5. Проверка датасета и реконструкции

```bash
lipsync data-validate --data-root ./data/my_grid_prepared
lipsync dataset-verify --checkpoint checkpoints/best.pt --data-root ./data/my_grid_prepared --split val --num-clips 3
```

## 6. API Python (пакетный инференс)

Минимальный вызов из кода — см. [`python_apply_batch.py`](python_apply_batch.py).
