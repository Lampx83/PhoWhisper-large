"""
PhoWhisper ASR API — Vietnamese speech to text.
Default model: vinai/PhoWhisper-small (16 kHz mono). Override: PHOWHISPER_MODEL.
"""

import io
import logging
import os
import threading
import time
from typing import Any

import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("PHOWHISPER_MODEL", "vinai/PhoWhisper-small")
TARGET_SR = 16000
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
# Log định kỳ trong lúc ASR (xem progress: docker logs -f <container>)
ASR_HEARTBEAT_SEC = float(os.environ.get("ASR_HEARTBEAT_SEC", "45"))
# Whisper: tên ngôn ngữ theo model (bỏ qua bước detect, thường nhanh hơn). Để trống = tự detect.
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "vietnamese").strip()

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
        device, compute_dtype = _device_and_dtype()
        logger.info("Loading model %s on device=%s dtype=%s", MODEL_ID, device, compute_dtype)
        from transformers import pipeline

        _transcriber = pipeline(
            "automatic-speech-recognition",
            model=MODEL_ID,
            dtype=compute_dtype,
            device=device,
        )
        logger.info("Model ready")
    return _transcriber


app = FastAPI(
    title="PhoWhisper ASR API",
    description="Chuyển âm thanh tiếng Việt sang văn bản (mặc định PhoWhisper-small).",
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

    duration_s = float(len(audio)) / TARGET_SR
    logger.info(
        "Transcribe start: file=%s size=%dB duration=%.1fs (~%.1f min audio) model=%s — "
        "Dung lượng MB nhỏ vẫn có thể là file dài (nén). Trên CPU: thời xử lý thường >> vài phút với >10–20 phút âm thanh. "
        "curl không có progress; xem docker logs -f.",
        file.filename,
        len(body),
        duration_s,
        duration_s / 60.0,
        MODEL_ID,
    )

    stop_hb = threading.Event()

    def _heartbeat():
        elapsed = 0.0
        while not stop_hb.wait(timeout=ASR_HEARTBEAT_SEC):
            elapsed += ASR_HEARTBEAT_SEC
            logger.info("ASR đang chạy… ~%.0f s", elapsed)

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()
    t0 = time.perf_counter()
    try:
        pipe = get_transcriber()
        # >~30s: bắt buộc return_timestamps. Không dùng chunk_length_s trên seq2seq — HF khuyên để Whisper tự chunk (generate).
        gen_kw: dict[str, Any] = {"task": "transcribe"}
        if ASR_LANGUAGE:
            gen_kw["language"] = ASR_LANGUAGE
        out = pipe(
            {"array": audio.astype(np.float32), "sampling_rate": TARGET_SR},
            return_timestamps=True,
            generate_kwargs=gen_kw,
        )
    except Exception as e:
        logger.exception("Transcribe error: %s", e)
        raise HTTPException(500, "Lỗi khi nhận dạng giọng nói") from e
    finally:
        stop_hb.set()
        hb.join(timeout=2.0)

    logger.info("Transcribe xong trong %.1f s (audio %.1f s)", time.perf_counter() - t0, duration_s)

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
