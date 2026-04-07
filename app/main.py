"""
PhoWhisper-large ASR API — Vietnamese speech to text.
Model: https://huggingface.co/vinai/PhoWhisper-large (16 kHz mono input).
"""

import io
import logging
import os
import threading
from typing import Any

import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("PHOWHISPER_MODEL", "vinai/PhoWhisper-large")
TARGET_SR = 16000
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))

_transcriber: Any = None
_load_lock = threading.Lock()


def _device_and_dtype() -> tuple:
    if torch.cuda.is_available():
        return 0, torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float32
    return -1, torch.float32


def get_transcriber():
    global _transcriber
    if _transcriber is not None:
        return _transcriber
    with _load_lock:
        if _transcriber is not None:
            return _transcriber
        device, torch_dtype = _device_and_dtype()
        logger.info("Loading model %s on device=%s dtype=%s", MODEL_ID, device, torch_dtype)
        from transformers import pipeline

        _transcriber = pipeline(
            "automatic-speech-recognition",
            model=MODEL_ID,
            torch_dtype=torch_dtype,
            device=device,
        )
        logger.info("Model ready")
    return _transcriber


app = FastAPI(
    title="PhoWhisper ASR API",
    description="Chuyển âm thanh tiếng Việt sang văn bản (PhoWhisper-large).",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscribeResponse(BaseModel):
    text: str = Field(..., description="Bản ghi chữ từ âm thanh")
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


@app.on_event("startup")
def startup_load_model():
    if os.environ.get("PRELOAD_MODEL", "1").lower() in ("1", "true", "yes"):
        try:
            get_transcriber()
        except Exception as e:
            logger.exception("Preload failed (sẽ thử lại khi có request): %s", e)


@app.get("/health", response_model=HealthResponse)
def health():
    dev, _ = _device_and_dtype()
    dev_str = str(dev)
    if _transcriber is None:
        return HealthResponse(status="starting", model=MODEL_ID, device=dev_str)
    return HealthResponse(status="ok", model=MODEL_ID, device=dev_str)


@app.post("/v1/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(..., description="File âm thanh (wav, mp3, m4a, flac, …)")):
    if not file.filename:
        raise HTTPException(400, "Thiếu tên file")
    body = await file.read()
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(body) > max_bytes:
        raise HTTPException(413, f"File quá lớn (tối đa {MAX_UPLOAD_MB} MB)")

    try:
        audio, _ = librosa.load(io.BytesIO(body), sr=TARGET_SR, mono=True)
    except Exception as e:
        logger.warning("Không đọc được audio: %s", e)
        raise HTTPException(400, "Không đọc được file âm thanh. Thử wav/mp3/flac.") from e

    if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-6:
        raise HTTPException(400, "Âm thanh rỗng hoặc quá nhỏ")

    try:
        pipe = get_transcriber()
        out = pipe({"array": audio.astype(np.float32), "sampling_rate": TARGET_SR})
    except Exception as e:
        logger.exception("Transcribe error: %s", e)
        raise HTTPException(500, "Lỗi khi nhận dạng giọng nói") from e

    text = (out.get("text") or "").strip()
    return TranscribeResponse(text=text, model=MODEL_ID)


@app.get("/")
def root():
    return {
        "service": "phowhisper-asr",
        "docs": "/docs",
        "transcribe": "POST /v1/transcribe (multipart: file)",
        "health": "/health",
    }
