# PhoWhisper ASR API

API HTTP chuyển **âm thanh tiếng Việt → văn bản**. **Mặc định:** [vinai/PhoWhisper-small](https://huggingface.co/vinai/PhoWhisper-small) (16 kHz, mono). Đổi model qua biến `PHOWHISPER_MODEL` (ví dụ `vinai/PhoWhisper-large`).

## Chạy local (Python)

```bash
cd PhoWhisper-large
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch  # máy không GPU: có thể dùng pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# tùy chọn: export PHOWHISPER_MODEL=vinai/PhoWhisper-large
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Mở [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Docker

```bash
docker compose up --build
```

- Lần đầu container sẽ tải model từ Hugging Face (dung lượng tùy `small` / `large`) — cần mạng ổn định.
- Volume `hf_cache` giữ cache model giữa các lần restart.
- API trên host: cổng **8023** (ví dụ `http://<host>:8023/docs`).

## Portainer

1. Push repo lên GitHub.
2. Trong Portainer: **Stacks** → **Add stack** → dán nội dung `docker-compose.yml` (hoặc Git repository URL nếu bạn bật deploy từ repo).
3. **Environment variables** (tùy chọn): `PHOWHISPER_MODEL`, `MAX_UPLOAD_MB`, `CORS_ORIGINS`, `HF_TOKEN` (nếu sau này dùng model private).
4. **RAM:** mặc định `small` thường **≥ 2–4 GB**; `PhoWhisper-large` trên CPU nên **≥ 8 GB**. Chỉnh `deploy.resources` trong compose nếu cần.

**Lỗi `pull access denied for phowhisper-asr`:** image chỉ được **build từ Dockerfile**, không có trên Docker Hub. Trong compose đã bỏ `image: phowhisper-asr:latest` và dùng `pull_policy: build`. Trên Portainer hãy **Update the stack** từ Git rồi deploy kiểu **build lại** (không chỉ “Pull” image). Nếu Portainer cũ không hỗ trợ `pull_policy`, xóa dòng đó trong compose vẫn ổn miễn là không có `image` trỏ tên Hub.

## API

| Method | Path | Mô tả |
|--------|------|--------|
| GET | `/health` | Trạng thái + thiết bị |
| POST | `/v1/transcribe` | `multipart/form-data`, field `file`: wav/mp3/flac/… |
| POST | `/v1/transcribe/stream` | Cùng `file`, trả **SSE** (`text/event-stream`): từng đoạn ~`ASR_STREAM_CHUNK_SEC` giây (mặc định 60s) xong là một sự kiện `segment` — không cần chờ hết file. |

**Stream (curl):** bắt buộc `-N` (không buffer). Mỗi dòng `data: {...}` là JSON: `start` → nhiều `segment` (`time_start`, `time_end`, `text`) → `done` (`full_text`).

```bash
curl -N -sS -X POST "http://127.0.0.1:8023/v1/transcribe/stream" -F "file=@sample.mp3"
```

**Chờ lâu, curl không có progress:** thời gian phụ thuộc **độ dài ghi âm (phút)**, không phải dung lượng MB (MP3 nén 10MB có thể ~20 phút thoại). Trên CPU, **~20 phút audio** với `small` dễ mất **hàng chục phút đến >1 giờ** xử lý. Xem log: `docker logs -f pho_whisper-phowhisper-api-1`. Timeout: `curl --max-time 7200 ...`.

Mặc định `ASR_LANGUAGE=vietnamese` (trong code) để bỏ bước detect ngôn ngữ. Tự detect: đặt biến môi trường `ASR_LANGUAGE` rỗng trên stack.

Ví dụ với **file mẫu tiếng Việt** trong repo (`samples/Pv14.m4a`, ~11 MB):

```bash
git clone https://github.com/Lampx83/PhoWhisper-large.git
cd PhoWhisper-large
curl -sS -X POST "http://127.0.0.1:8023/v1/transcribe" -F "file=@samples/Pv14.m4a"
```

Hoặc tải thẳng file mẫu (không clone cả repo):

```bash
curl -sSL -o Pv14.m4a https://raw.githubusercontent.com/Lampx83/PhoWhisper-large/main/samples/Pv14.m4a
curl -sS -X POST "http://127.0.0.1:8023/v1/transcribe" -F "file=@Pv14.m4a"
```

### Chưa có file âm thanh (tone thử API)

Trên máy chủ chỉ cần **Python 3** (không cài thêm gói):

```bash
curl -sS -o make_test_wav.py https://raw.githubusercontent.com/Lampx83/PhoWhisper-large/main/scripts/make_test_wav.py
python3 make_test_wav.py -o test_16k.wav
curl -sS -X POST "http://127.0.0.1:8023/v1/transcribe" -F "file=@test_16k.wav"
```

Tone 2 giây không phải tiếng Việt — chỉ để kiểm tra API chạy.

Nếu đã có **ffmpeg** trên server:

```bash
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" -ar 16000 -ac 1 test_16k.wav
```

## Máy chủ mạnh (ví dụ 256GB RAM, 64 CPU, GPU 48GB)

**Mục tiêu:** để inference ASR chủ yếu chạy trên **GPU** (nhanh nhất), còn **CPU** phục vụ librosa/numpy và các op PyTorch fallback; **RAM** để cache model + file dài.

### 1. Docker + NVIDIA

- Cài **NVIDIA driver** + **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)** trên host.
- Kiểm tra: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`

### 2. Build & chạy stack GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

- Dùng **`Dockerfile.gpu`** (image `pytorch/pytorch` + CUDA). Nếu pull lỗi tag, sửa `ARG PYTORCH_IMAGE` trong `Dockerfile.gpu` cho khớp driver (xem [tags PyTorch](https://hub.docker.com/r/pytorch/pytorch/tags)).
- Compose GPU file đặt mặc định **`PHOWHISPER_MODEL=vinai/PhoWhisper-large`**, giới hạn RAM **~200G**, **`gpus: all`**.

### 3. Tận dụng 64 core CPU (không nhân đôi GPU)

- **Một tiến trình / một worker** (`uvicorn --workers 1`): một bản model trên GPU — tránh ăn VRAM gấp 2–4 lần vô ích.
- Tăng luồng **tính toán CPU** (preprocess, một số op):

  | Biến | Gợi ý (64 core) | Ý nghĩa |
  |------|------------------|---------|
  | `TORCH_NUM_INTRAOP_THREADS` | `24`–`32` | Op song song trong một forward |
  | `TORCH_NUM_INTEROP_THREADS` | `2`–`4` | Song song giữa các op độc lập |
  | `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` | `24`–`32` | numpy/librosa/OpenBLAS |

- Tuning: tăng dần `TORCH_NUM_INTRAOP_THREADS` và đo latency; quá cao đôi khi **chậm hơn** do overhead.

- **`NVIDIA_TF32_OVERRIDE=1`**: bật TF32 trên Ampere+ (thường nhanh hơn, đủ cho ASR).

### 4. Nhiều request đồng thời

- **Một GPU** thường xử lý **tuần tự** từng request (trừ khi tự batch). Muốn **throughput** cao:
  - chạy **nhiều replica** trên **nhiều GPU** (`CUDA_VISIBLE_DEVICES=0` / `1` + hai stack hoặc hai container cổng khác nhau), hoặc
  - dùng queue (Redis/Celery) + worker GPU, hoặc
  - sau này cân nhắc **Triton / batch inference** (không có sẵn trong repo này).

### 5. Kiểm tra đã dùng GPU

```bash
curl -s http://127.0.0.1:8023/health
```

Trường **`gpu`** có tên card và **`device`** thường là `0` khi CUDA OK.

### 6. Chạy CPU-only nhưng “hết core”

Giữ `docker-compose.yml` mặc định, đặt các biến luồng như trên trong `environment`. Có thể tăng **`--workers`** uvicorn (mỗi worker **một bản model trong RAM** — chỉ hợp lý khi RAM rất lớn và không dùng chung một GPU).

## GPU (tóm tắt)

- **CPU (mặc định):** `Dockerfile` + `docker-compose.yml`.
- **GPU:** `Dockerfile.gpu` + `docker-compose.gpu.yml` (xem mục “Máy chủ mạnh”).

## License

Tuân theo license của model PhoWhisper (BSD-3-Clause) và thư viện bạn sử dụng.
