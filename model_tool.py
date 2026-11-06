"""
모델 도구 (Windows 엔진용) — 모델 받기, 양자화해서 저장, 시험 전사, 그리고 양자화해서 불러오기(load_asr).

windows_app.py / talkit_win.py 가 엔진 Python 으로 실행한다. stdout 에 JSON 을 한 줄씩 쓴다.
    {"type": "progress", "done": 바이트, "total": 바이트}
    {"type": "log", "msg": "…"}
    {"type": "done", ...}    /    {"type": "error", "msg": "…"}

사용
    python model_tool.py download --repo TeamUNIVA/qwen3_asr_1.7b_ko --dir 폴더
    python model_tool.py quantize --src 폴더 --dst 새폴더 --bits 4      (NVIDIA GPU + bitsandbytes 필요)
    python model_tool.py test --model 폴더 --quant nf4 --wav a.wav       (속도·메모리·결과 확인)

양자화 방식
    int8 / nf4   : bitsandbytes 로 GPU 에서 8-bit / 4-bit. 불러올 때 바로 하거나, quantize 로 저장해 두고 씀
    cpu-int8     : PyTorch 동적 양자화(Linear 층만 int8). GPU 없는 PC 용, 불러올 때마다 함(몇 초)
"""

import argparse
import json
import os
import shutil
import sys
import threading
import time


def out(**msg):
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def dir_size(p):
    total = 0
    for root, _, files in os.walk(p):
        for n in files:
            try:
                total += os.path.getsize(os.path.join(root, n))
            except OSError:
                pass
    return total


def prequantized(model_path):
    """이미 양자화해서 저장한 폴더면 그 설정(dict)을, 아니면 None"""
    try:
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
            return json.load(f).get("quantization_config")
    except Exception:
        return None


def resolve_quant(quant):
    import torch

    if quant != "auto":
        return quant
    if not torch.cuda.is_available():
        return "cpu-int8"
    return "none" if torch.cuda.get_device_properties(0).total_memory >= 6e9 else "nf4"


def load_asr(model_path, quant="auto", device="auto", max_new_tokens=256):
    """(Qwen3ASRModel, 설명) — quant: auto / none / int8 / nf4 / cpu-int8"""
    import torch
    from qwen_asr import Qwen3ASRModel

    kw = dict(max_inference_batch_size=1, max_new_tokens=max_new_tokens)
    cuda = torch.cuda.is_available() and device != "cpu"
    pre = prequantized(model_path)
    if pre:
        if not cuda:
            raise RuntimeError("이 모델은 GPU(bitsandbytes)로 양자화해 저장한 것이라 NVIDIA GPU 에서만 불러올 수 있어요.")
        asr = Qwen3ASRModel.from_pretrained(model_path, device_map="cuda:0", dtype=torch.bfloat16, **kw)
        return asr, f"cuda:0 · 저장된 {'4' if pre.get('load_in_4bit') else '8'}-bit"

    quant = resolve_quant(quant) if device != "cpu" else ("cpu-int8" if quant in ("auto", "cpu-int8") else "none")
    if quant in ("int8", "nf4"):
        if not cuda:
            raise RuntimeError(f"{quant} 양자화는 NVIDIA GPU 가 필요해요. GPU 없는 PC 는 'CPU 8-bit' 를 고르세요.")
        from transformers import BitsAndBytesConfig

        qc = (BitsAndBytesConfig(load_in_8bit=True) if quant == "int8" else
              BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
        asr = Qwen3ASRModel.from_pretrained(model_path, quantization_config=qc, device_map="cuda:0",
                                            dtype=torch.bfloat16, **kw)
        return asr, f"cuda:0 · {'8' if quant == 'int8' else '4'}-bit"
    if quant == "cpu-int8":
        asr = Qwen3ASRModel.from_pretrained(model_path, device_map="cpu", dtype=torch.float32, **kw)
        # 일부 층이 bf16 으로 올라와서, 양자화 뒤 model.dtype 이 bf16 으로 보이고 입력이 bf16 으로 바뀌어 실패한다 → 먼저 fp32 로 통일
        asr.model = torch.ao.quantization.quantize_dynamic(asr.model.float(), {torch.nn.Linear}, dtype=torch.qint8)
        asr.dtype = torch.float32
        return asr, "cpu · 8-bit(동적)"
    if cuda:
        return Qwen3ASRModel.from_pretrained(model_path, device_map="cuda:0", dtype=torch.bfloat16, **kw), "cuda:0 · bf16"
    return Qwen3ASRModel.from_pretrained(model_path, device_map="cpu", dtype=torch.float32, **kw), "cpu · fp32"


# ---------------------------------------------------------------- 명령
# 전사에 안 쓰는 다른 프레임워크용 무게 파일은 받지 않는다
SKIP = ["*.h5", "*.onnx", "onnx/*", "*.msgpack", "*.ot", "*.tflite", "*.gguf"]


def cmd_download(a):
    import fnmatch

    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(a.repo, files_metadata=True)
    names = [s.rfilename for s in info.siblings]
    skip = list(SKIP)
    if any(n.endswith(".safetensors") for n in names):
        skip += ["*.bin", "*.pt", "*.pth"]  # safetensors 가 있으면 옛 형식은 필요 없음
    total = sum((s.size or 0) for s in info.siblings if not any(fnmatch.fnmatch(s.rfilename, p) for p in skip))
    os.makedirs(a.dir, exist_ok=True)
    out(type="log", msg=f"{a.repo} 받는 중 ({total / 1e9:.1f}GB) → {a.dir}")
    done = threading.Event()

    def report():
        while not done.wait(1.0):
            out(type="progress", done=dir_size(a.dir), total=total)

    threading.Thread(target=report, daemon=True).start()
    try:
        snapshot_download(repo_id=a.repo, local_dir=a.dir, ignore_patterns=skip)
    finally:
        done.set()
    shutil.rmtree(os.path.join(a.dir, ".cache"), ignore_errors=True)  # 받는 중에 쓰던 임시 폴더
    out(type="progress", done=total, total=total)
    out(type="done", dir=a.dir, size=dir_size(a.dir))


def cmd_quantize(a):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("양자화해서 저장하기는 NVIDIA GPU 가 필요해요 (bitsandbytes).")
    if os.path.exists(a.dst):
        raise RuntimeError(f"이미 있는 폴더예요: {a.dst}")
    t = time.time()
    out(type="log", msg=f"{a.bits}-bit 로 불러오는 중… ({a.src})")
    asr, desc = load_asr(a.src, "nf4" if a.bits == 4 else "int8")
    tmp = a.dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    # 토크나이저·전처리 설정 같은 작은 파일은 그대로 복사하고, 무게(가중치)와 config 는 새로 저장한다
    for n in os.listdir(a.src):
        p = os.path.join(a.src, n)
        if os.path.isfile(p) and not n.endswith(".safetensors") and not n.endswith(".index.json"):
            shutil.copy2(p, os.path.join(tmp, n))
    out(type="log", msg="저장 중…")
    asr.model.save_pretrained(tmp, safe_serialization=True)
    os.replace(tmp, a.dst)
    out(type="done", dir=a.dst, size=dir_size(a.dst), src_size=dir_size(a.src), sec=round(time.time() - t, 1),
        desc=desc)


def cmd_test(a):
    import numpy as np
    import soundfile as sf
    import torch

    t = time.time()
    asr, desc = load_asr(a.model, a.quant, a.device)
    load_sec = time.time() - t
    audio, sr = sf.read(a.wav, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    asr.transcribe(audio=(np.zeros(16000, np.float32), 16000), language="Korean")  # 데우기
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    res = asr.transcribe(audio=(audio, sr), language="Korean")
    sec = time.time() - t
    mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() and "cuda" in desc else None
    if mem is None:
        try:
            import psutil
            mem = psutil.Process().memory_info().rss / 1e9
        except Exception:
            pass
    out(type="done", desc=desc, load_sec=round(load_sec, 1), audio_sec=round(len(audio) / sr, 1),
        infer_sec=round(sec, 2), mem_gb=round(mem, 2) if mem else None, text=res[0].text)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--repo", required=True)
    d.add_argument("--dir", required=True)
    q = sub.add_parser("quantize")
    q.add_argument("--src", required=True)
    q.add_argument("--dst", required=True)
    q.add_argument("--bits", type=int, choices=[4, 8], default=4)
    t = sub.add_parser("test")
    t.add_argument("--model", required=True)
    t.add_argument("--quant", default="auto")
    t.add_argument("--device", default="auto")
    t.add_argument("--wav", required=True)
    a = p.parse_args()
    try:
        {"download": cmd_download, "quantize": cmd_quantize, "test": cmd_test}[a.cmd](a)
    except Exception as e:
        out(type="error", msg=f"{type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    # 라이브러리가 stdout 에 찍는 글이 JSON 줄과 섞이지 않게 stderr 로 돌린다
    _real = sys.stdout
    sys.stdout = sys.stderr
    _out = out

    def out(**msg):  # noqa: F811
        _real.write(json.dumps(msg, ensure_ascii=False) + "\n")
        _real.flush()

    sys.exit(main())
