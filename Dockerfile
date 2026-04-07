# CPU mặc định — đủ cho triển khai Portainer thông thường.
# GPU: dùng image nvidia/cuda + cài torch CUDA (xem README).
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    PHOWHISPER_MODEL=vinai/PhoWhisper-large \
    PRELOAD_MODEL=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# PyTorch CPU — tránh bản CUDA (nvidia-*) trong container không GPU
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
