"""FastAPI application for lip-sync inference via REST and WebSocket."""
from __future__ import annotations

import asyncio
import base64
import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..runtime import assert_runtime_compatible

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

_pipeline = None
_batch_processor = None


def create_app(
    checkpoint_path: str | Path,
    device: str = "auto",
    use_sr: bool = False,
    sr_backend: str = "gfpgan",
    fps: float = 25.0,
    audio_window_ms: float = 200.0,
) -> Any:
    """Build and return the FastAPI application.

    Args:
        checkpoint_path: Path to trained model weights.
        device: Inference device.
        use_sr: Enable super-resolution.
        sr_backend: SR backend name.
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError("fastapi and uvicorn are required. Run: pip install fastapi uvicorn")

    from ..inference.batch.processor import BatchProcessor
    from ..inference.realtime.pipeline import RealTimePipeline

    global _pipeline, _batch_processor

    _pipeline = RealTimePipeline(
        checkpoint_path=checkpoint_path,
        device=device,
        use_sr=use_sr,
        sr_backend=sr_backend,
        fps=fps,
        audio_window_ms=audio_window_ms,
    )
    _batch_processor = BatchProcessor(
        checkpoint_path=checkpoint_path,
        device=device,
        use_sr=use_sr,
        sr_backend=sr_backend,
    )

    assert_runtime_compatible(require_cv2=True, require_torchvision=False)

    app = FastAPI(
        title="LipSync Framework API",
        description="Real-time and batch audio-driven lip synchronisation.",
        version="1.0.0",
    )

    REQUEST_LIMITS = {
        "max_video_bytes": 300 * 1024 * 1024,
        "max_audio_bytes": 20 * 1024 * 1024,
        "max_frame_bytes": 5 * 1024 * 1024,
        "max_ws_queue": 8,
    }
    ws_semaphore = asyncio.Semaphore(REQUEST_LIMITS["max_ws_queue"])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    @app.post("/lipsync/batch")
    async def lipsync_batch(
        video: UploadFile = File(...),
        audio: UploadFile = File(...),
    ):
        """Drive a video with an audio file and return the result MP4."""
        import cv2

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            video_path = tmp / video.filename
            audio_path = tmp / audio.filename
            output_path = tmp / "output.mp4"

            video_bytes = await video.read()
            audio_bytes = await audio.read()
            if len(video_bytes) > REQUEST_LIMITS["max_video_bytes"]:
                raise HTTPException(status_code=413, detail="Video too large")
            if len(audio_bytes) > REQUEST_LIMITS["max_audio_bytes"]:
                raise HTTPException(status_code=413, detail="Audio too large")

            video_path.write_bytes(video_bytes)
            audio_path.write_bytes(audio_bytes)

            _batch_processor.process(
                video_path=video_path,
                audio_path=audio_path,
                output_path=output_path,
            )

            if not output_path.exists():
                raise HTTPException(status_code=500, detail="Processing failed")

            return FileResponse(
                str(output_path),
                media_type="video/mp4",
                filename="lipsync_output.mp4",
            )

    # ------------------------------------------------------------------
    # Single-frame inference (for streaming integration)
    # ------------------------------------------------------------------

    @app.post("/lipsync/frame")
    async def lipsync_frame(
        frame_b64: str = Form(...),
        audio_b64: str = Form(...),
    ):
        """Process a single frame with an audio chunk.

        Both frame and audio are Base64-encoded.

        Returns:
            JSON with 'frame_b64' key containing the processed frame.
        """
        import cv2

        frame_bytes = base64.b64decode(frame_b64)
        if len(frame_bytes) > REQUEST_LIMITS["max_frame_bytes"]:
            raise HTTPException(status_code=413, detail="Frame too large")
        frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

        audio_bytes = base64.b64decode(audio_b64)
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        _pipeline.push_audio(audio_np)

        out_frame = _pipeline.process_frame(frame)

        _, enc = cv2.imencode(".jpg", out_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out_b64 = base64.b64encode(enc.tobytes()).decode("utf-8")
        return JSONResponse({"frame_b64": out_b64})

    # ------------------------------------------------------------------
    # WebSocket streaming
    # ------------------------------------------------------------------

    @app.websocket("/ws/lipsync")
    async def lipsync_ws(websocket: WebSocket):
        """WebSocket endpoint for real-time lip-sync streaming.

        Message protocol (JSON):
        - Client → Server: {"type": "audio", "data": "<base64 PCM float32>"}
        - Client → Server: {"type": "frame", "data": "<base64 JPEG>"}
        - Server → Client: {"type": "frame", "data": "<base64 JPEG>"}
        """
        import cv2

        await websocket.accept()
        try:
            while True:
                async with ws_semaphore:
                    msg = await websocket.receive_json()
                mtype = msg.get("type")
                data = msg.get("data", "")

                if mtype == "audio":
                    raw = np.frombuffer(base64.b64decode(data), dtype=np.float32)
                    _pipeline.push_audio(raw)

                elif mtype == "frame":
                    frame_bytes = base64.b64decode(data)
                    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                    out = _pipeline.process_frame(frame)
                    _, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    await websocket.send_json(
                        {"type": "frame", "data": base64.b64encode(enc.tobytes()).decode()}
                    )

                elif mtype == "set_reference":
                    frame_bytes = base64.b64decode(data)
                    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                    _pipeline.set_reference_frame(frame)
                    await websocket.send_json({"type": "ack", "msg": "reference set"})

        except WebSocketDisconnect:
            pass

    return app


def run_server(
    checkpoint_path: str | Path,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "auto",
    use_sr: bool = False,
    sr_backend: str = "gfpgan",
    fps: float = 25.0,
    audio_window_ms: float = 200.0,
    **kwargs,
) -> None:
    """Start the uvicorn server."""
    import uvicorn

    app = create_app(
        checkpoint_path=checkpoint_path,
        device=device,
        use_sr=use_sr,
        sr_backend=sr_backend,
        fps=fps,
        audio_window_ms=audio_window_ms,
    )
    uvicorn.run(app, host=host, port=port, **kwargs)
