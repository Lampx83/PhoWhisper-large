#!/usr/bin/env python3
"""Tạo file WAV 16 kHz mono (mặc định: tone 2s) để thử POST /v1/transcribe — chỉ dùng thư viện chuẩn Python."""
import argparse
import math
import struct
import wave


def main():
    p = argparse.ArgumentParser(description="Tạo WAV thử /v1/transcribe")
    p.add_argument("-o", "--output", default="test_16k.wav", help="Đường dẫn file .wav ghi ra")
    p.add_argument("-d", "--duration", type=float, default=2.0, help="Độ dài (giây)")
    p.add_argument(
        "--silent",
        action="store_true",
        help="Toàn im lặng (API sẽ trả 400 vì biên độ quá nhỏ)",
    )
    args = p.parse_args()

    rate = 16000
    n = int(rate * args.duration)
    with wave.open(args.output, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        for i in range(n):
            if args.silent:
                s = 0
            else:
                # Tone 440 Hz — đủ biên độ để /v1/transcribe chấp nhận (không phải tiếng Việt)
                s = int(32767 * 0.2 * math.sin(2 * math.pi * 440 * i / rate))
            w.writeframes(struct.pack("<h", max(-32768, min(32767, s))))

    print(f"Đã tạo {args.output} ({args.duration}s, 16kHz mono)")


if __name__ == "__main__":
    main()
