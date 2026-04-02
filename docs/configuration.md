# Конфигурация

Базовый файл: [`configs/base.yaml`](../configs/base.yaml). Его можно передать в `train`:

```bash
lipsync train --config configs/base.yaml --data-root ./data/prepared
```

## Основные секции YAML

| Секция | Назначение |
|--------|------------|
| `model` | Размерности Conformer, ResNet-identity, U-Net, дискриминатор, SyncNet |
| `audio` | `sample_rate`, `n_mels`, длина окна мелов (`window`) |
| `video` | `face_size`, `lip_size`, `target_fps` |
| `lipsync` | `sync_window`, `temporal_radius`, вес области рта |
| `inference` | Сглаживание, вставка рта, mux аудио в MP4, **композитинг губ** |
| `optimizer` / `scheduler` | Обучение |
| `losses` | Веса и параметры лоссов |
| `data` | Пути, аугментации, `static_face_prob` и т.д. |

## Важные поля `inference` (качество и паритет train/infer)

- **`audio_embed_pool`**: `last` (рекомендуется, выравнивание по кадру) или `mean`.
- **`mouth_composite_mode`**: `blend` (мягкая вставка) vs `paste` / `hard_lower`.
- **`mux_driving_audio`**: подмешать driving WAV в выходной MP4 через ffmpeg.
- **`keep_original_audio`**: сохранять исходную дорожку видео там, где применимо.

Подробные комментарии — прямо в `configs/base.yaml`.

## Профили рантайма (`train --profile`)

| Профиль | Идея |
|---------|------|
| `cpu-safe` | Меньше нагрузка, стабильность на CPU |
| `gpu-fast` | Ускорение на GPU |
| `gpu-quality` | Более тяжёлый режим в пользу качества |

Профили подмешиваются поверх YAML через `lipsync.runtime.apply_profile_to_config`.

## Дополнительные конфиги

В репозитории могут быть варианты (например, `configs/synthetic_dataset_only.yaml`) для экспериментов на синтетике — смотрите комментарии внутри файла.
