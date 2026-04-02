"""Программный пакетный липсинк через API фреймворка.

Запуск из корня репозитория (после `pip install -e .`):

    python examples/python_apply_batch.py

Перед запуском задайте пути к чекпоинту, видео и WAV.
"""
from __future__ import annotations

from pathlib import Path

from lipsync import apply_batch

# --- замените на свои файлы ---
CHECKPOINT = Path("checkpoints/best.pt")
VIDEO = Path("input.mp4")
AUDIO = Path("speech.wav")
OUTPUT = Path("output_programmatic.mp4")


def main() -> None:
    out = apply_batch(
        checkpoint=CHECKPOINT,
        video=VIDEO,
        audio=AUDIO,
        output=OUTPUT,
        device="auto",
        fps=25.0,
    )
    print("Written:", out.resolve())


if __name__ == "__main__":
    main()
