# PhoWhisper-large ASR API

API HTTP chuyển **âm thanh tiếng Việt → văn bản**, dùng model [vinai/PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large) (16 kHz, mono).

## Chạy local (Python)

```bash
cd PhoWhisper-large
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch  # máy không GPU: có thể dùng pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export PHOWHISPER_MODEL=vinai/PhoWhisper-large
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Mở [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Docker

```bash
docker compose up --build
```

- Lần đầu container sẽ tải model từ Hugging Face (vài GB) — cần mạng ổn định.
- Volume `hf_cache` giữ cache model giữa các lần restart.
- API trên host: cổng **8023** (ví dụ `http://<host>:8023/docs`).

## Portainer

1. Push repo lên GitHub.
2. Trong Portainer: **Stacks** → **Add stack** → dán nội dung `docker-compose.yml` (hoặc Git repository URL nếu bạn bật deploy từ repo).
3. **Environment variables** (tùy chọn): `PHOWHISPER_MODEL`, `MAX_UPLOAD_MB`, `CORS_ORIGINS`, `HF_TOKEN` (nếu sau này dùng model private).
4. Cấp **RAM đủ lớn** (khuyến nghị **≥ 8 GB** cho `large` trên CPU). Nếu OOM, hạ model xuống `vinai/PhoWhisper-medium` qua biến `PHOWHISPER_MODEL`.

## API

| Method | Path | Mô tả |
|--------|------|--------|
| GET | `/health` | Trạng thái + thiết bị |
| POST | `/v1/transcribe` | `multipart/form-data`, field `file`: wav/mp3/flac/… |

Ví dụ:

```bash
curl -s -X POST "http://localhost:8023/v1/transcribe" -F "file=@sample.wav"
```

## GPU (tùy chọn)

Image mặc định dùng **PyTorch CPU**. Để chạy GPU trên máy có NVIDIA + nvidia-container-toolkit, cần base image CUDA và cài `torch` bản CUDA — có thể tách thêm `Dockerfile.gpu` sau nếu bạn cần.

## License

Tuân theo license của model PhoWhisper (BSD-3-Clause) và thư viện bạn sử dụng.
