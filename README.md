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

**Chờ lâu, curl không có progress:** trên CPU, file dài vẫn có thể mất nhiều phút (đặc biệt nếu đổi sang `large`). Xem log: `docker logs -f pho_whisper-phowhisper-api-1` (mỗi ~45s có dòng `ASR đang chạy…`). Đo thời gian: `date; curl ...; date`. Timeout: `curl --max-time 3600 ...`.

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

## GPU (tùy chọn)

Image mặc định dùng **PyTorch CPU**. Để chạy GPU trên máy có NVIDIA + nvidia-container-toolkit, cần base image CUDA và cài `torch` bản CUDA — có thể tách thêm `Dockerfile.gpu` sau nếu bạn cần.

## License

Tuân theo license của model PhoWhisper (BSD-3-Clause) và thư viện bạn sử dụng.
