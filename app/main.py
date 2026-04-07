"""
PhoWhisper ASR API — Vietnamese speech to text.
Default model: vinai/PhoWhisper-small (16 kHz mono). Override: PHOWHISPER_MODEL.
"""

import io
import json
import logging
import os
import threading
import time
from typing import Any, Iterator

import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
# Độ dài mỗi đoạn stream (giây) — endpoint /v1/transcribe/stream
ASR_STREAM_CHUNK_SEC = float(os.environ.get("ASR_STREAM_CHUNK_SEC", "60"))

_transcriber: Any = None
_load_lock = threading.Lock()


def _configure_compute_environment() -> None:
    """Đọc biến môi trường để dùng nhiều core CPU cho phần tính toán của PyTorch/OpenMP (librosa/numpy)."""
    intra = os.environ.get("TORCH_NUM_INTRAOP_THREADS")
    inter = os.environ.get("TORCH_NUM_INTEROP_THREADS")
    if intra:
        try:
            torch.set_num_threads(max(1, int(intra)))
        except ValueError:
            pass
    if inter:
        try:
            torch.set_num_interop_threads(max(1, int(inter)))
        except (ValueError, RuntimeError):
            pass
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if key in os.environ:
            logger.info("%s=%s", key, os.environ[key])
    logger.info(
        "Compute: torch.intra_threads=%s inter_threads=%s cuda_available=%s",
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
        torch.cuda.is_available(),
    )


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


def _generate_kwargs() -> dict[str, Any]:
    gen_kw: dict[str, Any] = {"task": "transcribe"}
    if ASR_LANGUAGE:
        gen_kw["language"] = ASR_LANGUAGE
    return gen_kw


async def _load_audio_from_upload(file: UploadFile) -> tuple[np.ndarray, str, int]:
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
    return audio, file.filename, len(body)


def _sse_line(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


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
    gpu: str | None = Field(default=None, description="Tên GPU (CUDA) nếu có")


def _preload_model_background():
    """Không chạy trong startup đồng bộ — Uvicorn/Starlette chỉ mở cổng sau khi startup xong; preload lâu làm /docs treo."""
    try:
        get_transcriber()
    except Exception as e:
        logger.exception("Preload nền thất bại (sẽ thử khi có request): %s", e)


@app.on_event("startup")
def startup_load_model():
    _configure_compute_environment()
    if os.environ.get("PRELOAD_MODEL", "1").lower() in ("1", "true", "yes"):
        threading.Thread(target=_preload_model_background, name="model-preload", daemon=True).start()
        logger.info(
            "Preload model chạy nền — /docs và /health phản hồi ngay; transcribe sau khi log 'Model ready'."
        )


@app.get("/health", response_model=HealthResponse)
def health():
    dev, _ = _device_and_dtype()
    dev_str = str(dev)
    gpu_name: str | None = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "cuda:0"
    if _transcriber is None:
        return HealthResponse(status="starting", model=MODEL_ID, device=dev_str, gpu=gpu_name)
    return HealthResponse(status="ok", model=MODEL_ID, device=dev_str, gpu=gpu_name)


@app.post("/v1/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(..., description="File âm thanh (wav, mp3, m4a, flac, …)")):
    audio, filename, body_len = await _load_audio_from_upload(file)

    duration_s = float(len(audio)) / TARGET_SR
    logger.info(
        "Transcribe start: file=%s size=%dB duration=%.1fs (~%.1f min audio) model=%s — "
        "Dung lượng MB nhỏ vẫn có thể là file dài (nén). Trên CPU: thời xử lý thường >> vài phút với >10–20 phút âm thanh. "
        "curl không có progress; xem docker logs -f.",
        filename,
        body_len,
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
        out = pipe(
            {"array": audio.astype(np.float32), "sampling_rate": TARGET_SR},
            return_timestamps=True,
            generate_kwargs=_generate_kwargs(),
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


def _stream_transcribe_events(audio: np.ndarray, filename: str, body_len: int) -> Iterator[str]:
    duration_s = float(len(audio)) / TARGET_SR
    chunk_samples = max(int(ASR_STREAM_CHUNK_SEC * TARGET_SR), TARGET_SR)  # tối thiểu ~1s
    logger.info(
        "Transcribe stream: file=%s size=%dB duration=%.1fs chunk=%.0fs model=%s",
        filename,
        body_len,
        duration_s,
        ASR_STREAM_CHUNK_SEC,
        MODEL_ID,
    )
    yield _sse_line(
        {
            "type": "start",
            "filename": filename,
            "duration_s": round(duration_s, 2),
            "chunk_sec": ASR_STREAM_CHUNK_SEC,
            "model": MODEL_ID,
        }
    )

    pipe = get_transcriber()
    gen_kw = _generate_kwargs()
    n = len(audio)
    start = 0
    idx = 0
    parts: list[str] = []
    t_wall0 = time.perf_counter()

    while start < n:
        end = min(start + chunk_samples, n)
        chunk = audio[start:end]
        if len(chunk) < int(0.2 * TARGET_SR):
            break
        t0 = start / TARGET_SR
        t1 = end / TARGET_SR
        try:
            out = pipe(
                {"array": chunk.astype(np.float32), "sampling_rate": TARGET_SR},
                return_timestamps=True,
                generate_kwargs=gen_kw,
            )
        except Exception as e:
            logger.exception("Stream segment %d error: %s", idx, e)
            yield _sse_line({"type": "error", "segment": idx, "message": str(e)})
            return
        seg_text = (out.get("text") or "").strip()
        parts.append(seg_text)
        yield _sse_line(
            {
                "type": "segment",
                "index": idx,
                "time_start": round(t0, 2),
                "time_end": round(t1, 2),
                "text": seg_text,
            }
        )
        logger.info("Stream segment %d done [%.1fs–%.1fs] %.0f chars", idx, t0, t1, len(seg_text))
        start = end
        idx += 1

    full_text = " ".join(p for p in parts if p).strip()
    yield _sse_line(
        {
            "type": "done",
            "full_text": full_text,
            "model": MODEL_ID,
            "wall_seconds": round(time.perf_counter() - t_wall0, 2),
        }
    )
    logger.info("Transcribe stream xong: %d đoạn, %.1f s wall", idx, time.perf_counter() - t_wall0)


@app.post("/v1/transcribe/stream")
async def transcribe_stream(
    file: UploadFile = File(..., description="File âm thanh — trả SSE từng đoạn theo ASR_STREAM_CHUNK_SEC"),
):
    """
    Server-Sent Events: mỗi đoạn ~ASR_STREAM_CHUNK_SEC giây xong sẽ có một sự kiện `segment`.
    Client dùng curl `-N` hoặc EventSource; không chờ hết file mới có output.
    """
    audio, filename, body_len = await _load_audio_from_upload(file)
    return StreamingResponse(
        _stream_transcribe_events(audio, filename, body_len),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
def root():
    return {
        "service": "phowhisper-asr",
        "docs": "/docs",
        "transcribe": "POST /v1/transcribe (multipart: file)",
        "transcribe_stream": "POST /v1/transcribe/stream (SSE, curl -N)",
        "health": "/health",
        "gpu_deploy": "compose -f docker-compose.yml -f docker-compose.gpu.yml (see README)",
    }
