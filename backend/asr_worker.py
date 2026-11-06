"""
전사 워커 — Rust 백엔드(asr-backend)가 WSL2 안에서 띄운다. 직접 실행할 일은 없다.

stdin : [u32 LE 샘플 수][f32 LE 샘플...]  (16kHz mono) 를 계속 받는다
stdout: 한 덩어리마다 JSON 한 줄  {"text": ..., "secs": ...} / {"error": ...}
        시작할 때 모델 로딩이 끝나면 {"ready": true, ...} 한 줄
로그   : stderr
"""

import argparse
import json
import os
import struct
import sys
import tempfile
import time

SAMPLE_RATE = 16000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="모델 폴더 경로 또는 HF 모델 ID")
    p.add_argument("--device", default="cuda:0", help="cuda:0 / cpu")
    args = p.parse_args()

    # stdout 은 백엔드와 주고받는 통로라서, 라이브러리가 찍는 print 는 stderr 로 돌린다
    out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    import numpy as np
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    t = time.time()
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device,
        max_inference_batch_size=1,
        max_new_tokens=256,
    )
    out.write(json.dumps({"ready": True, "device": args.device,
                          "load_sec": round(time.time() - t, 1)}) + "\n")

    stdin = sys.stdin.buffer
    # 공식 예제는 파일 경로 입력만 안내하므로 임시 wav 하나를 계속 덮어써서 넘긴다
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        while True:
            head = stdin.read(4)
            if len(head) < 4:
                break  # 백엔드 종료
            (n,) = struct.unpack("<I", head)
            data = stdin.read(n * 4)
            if len(data) < n * 4:
                break
            audio = np.frombuffer(data, dtype="<f4")

            t = time.time()
            try:
                sf.write(path, audio, SAMPLE_RATE)
                res = model.transcribe(audio=path)
                reply = {"text": (res[0].text or "").strip(), "secs": round(time.time() - t, 3)}
            except Exception as e:  # 한 덩어리 실패는 알리고 계속
                reply = {"error": f"{type(e).__name__}: {e}"}
            out.write(json.dumps(reply, ensure_ascii=False) + "\n")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
