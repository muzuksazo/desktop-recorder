"""
실시간 한국어 전사 (마이크 + 시스템 소리)
- Apple Silicon (Mac): `yeowon/qwen3_asr_1.7b_ko-MLX-4bit` (1.07GB, 4-bit 양자화, 실시간 스트리밍)
- Windows / Linux: `TeamUNIVA/qwen3_asr_1.7b_ko` (PyTorch CUDA / CPU)

동작 방식
- 마이크와 시스템 소리(스피커 출력)를 동시에 녹음해 섞는다.
- MLX 모드에서는 1초 단위로 실시간 스트리밍(`partial`) 중간 결과를 화면에 즉시 갱신하고,
  발화가 끝나거나 구간(`--chunk`, 기본 6초)이 차면 문장을 확정해 창과 `.txt` 파일에 저장한다.
"""

import argparse
import datetime as dt
import os
import platform
import queue
import sys
import tempfile
import threading

import numpy as np

# soundcard 패키지가 NumPy 2.x 에서 제거된 np.fromstring 이진 모드를 호출하는 호환성 패치
if not hasattr(np, "_orig_fromstring"):
    np._orig_fromstring = np.fromstring
    np.fromstring = lambda data, dtype=float, *a, **kw: np.frombuffer(data, dtype=dtype)

import soundcard as sc
import soundfile as sf

MLX_MODEL_ID = "yeowon/qwen3_asr_1.7b_ko-MLX-4bit"
PYTORCH_MODEL_ID = "TeamUNIVA/qwen3_asr_1.7b_ko"
MODEL_ID = MLX_MODEL_ID if (platform.system() == "Darwin" and platform.machine() == "arm64") else PYTORCH_MODEL_ID

SAMPLE_RATE = 16000
BLOCK_SEC = 0.5  # 녹음 스레드가 한 번에 읽는 길이


def _has_mlx() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
        import qwen3_asr_mlx  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- 오디오 장치
def find_system_audio_device(name_hint=None):
    """시스템 소리(루프백) 녹음 장치를 찾는다."""
    if name_hint:
        return sc.get_microphone(name_hint, include_loopback=True)

    if platform.system() == "Windows":
        return sc.get_microphone(sc.default_speaker().name, include_loopback=True)

    for m in sc.all_microphones(include_loopback=True):
        if "blackhole" in m.name.lower():
            return m
    return None


def list_devices():
    print("== 입력(마이크) 장치 ==")
    for m in sc.all_microphones(include_loopback=False):
        print("  ", m.name)
    print("== 루프백 포함 전체 ==")
    for m in sc.all_microphones(include_loopback=True):
        print("  ", m.name)
    print("== 기본 스피커 ==")
    print("  ", sc.default_speaker().name)


class Recorder(threading.Thread):
    """장치 하나에서 계속 녹음해서 out_q 에 0.5초 블록을 넣는다 (백프레셔 시 오래된 블록 드롭)."""

    def __init__(self, device, out_q, stop_event, label):
        super().__init__(daemon=True)
        self.device = device
        self.out_q = out_q
        self.stop_event = stop_event
        self.label = label

    def run(self):
        frames = int(SAMPLE_RATE * BLOCK_SEC)
        try:
            with self.device.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
                while not self.stop_event.is_set():
                    data = rec.record(numframes=frames)
                    arr = np.asarray(data, dtype=np.float32)
                    if arr.ndim > 1:
                        arr = arr[:, 0]
                    block = arr.reshape(-1)
                    try:
                        self.out_q.put_nowait(block)
                    except queue.Full:
                        try:
                            self.out_q.get_nowait()
                        except queue.Empty:
                            pass
                        self.out_q.put_nowait(block)
        except Exception as e:
            try:
                self.out_q.put_nowait(RuntimeError(f"[{self.label}] 녹음 오류: {e}"))
            except queue.Full:
                pass


# ---------------------------------------------------------------- 모델
def pick_device(requested):
    if requested == "mlx" or (requested == "auto" and _has_mlx()):
        return "mlx", "int4"

    import torch

    if requested != "auto":
        dev = requested
    elif torch.cuda.is_available():
        dev = "cuda:0"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"

    if dev.startswith("cuda"):
        dtype = torch.bfloat16
    elif dev == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return dev, dtype


def load_model(device_arg, model_id=None):
    dev, dtype = pick_device(device_arg)
    if dev == "mlx":
        from mlx_quantized_asr import QuantizedMLXQwen3ASR

        target_model = model_id or MLX_MODEL_ID
        print(f"MLX 4-bit 양자화 모델 로딩 중... (model={target_model})")
        engine = QuantizedMLXQwen3ASR(model_id_or_path=target_model, bits=4, group_size=64)
        engine.warm_up()
        print(f"MLX 4-bit 모델 로딩 완료 ({engine.total_mb:.1f} MB, {engine.load_time_sec:.2f}s)")
        return engine, f"mlx-4bit ({engine.total_mb:.0f}MB)"

    from qwen_asr import Qwen3ASRModel

    target_model = model_id or PYTORCH_MODEL_ID
    print(f"모델 로딩 중... (model={target_model}, device={dev}, dtype={dtype})")
    model = Qwen3ASRModel.from_pretrained(
        target_model,
        dtype=dtype,
        device_map=dev,
        max_inference_batch_size=1,
        max_new_tokens=256,
    )
    print("모델 로딩 완료")
    return model, dev


def transcribe_array(model, audio):
    """단일 청크 전사 (MLX 엔진과 PyTorch qwen-asr 모두 지원)."""
    if hasattr(model, "init_streaming_state") and hasattr(model, "asr") and hasattr(model, "total_mb"):
        res = model.transcribe(audio=audio, language="Korean")
        return (res.text or "").strip()

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(path, audio, SAMPLE_RATE)
        results = model.transcribe(audio=path)
        return (results[0].text or "").strip()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ---------------------------------------------------------------- 전사 루프
class Transcriber(threading.Thread):
    def __init__(self, args, ui_q, stop_event):
        super().__init__(daemon=True)
        self.args = args
        self.ui_q = ui_q
        self.stop_event = stop_event

    def run(self):
        args = self.args
        try:
            model, dev = load_model(args.device, getattr(args, "model", None))
        except Exception as e:
            self.ui_q.put(("error", f"모델 로딩 실패: {e}"))
            return

        is_mlx_streaming = dev.startswith("mlx")
        mode_desc = "실시간 1초 스트리밍" if is_mlx_streaming else f"{args.chunk}초 단위"
        self.ui_q.put(("status", f"듣는 중… (engine={dev}, {mode_desc})"))

        # 장치 준비
        sources = []
        if args.source in ("mic", "both"):
            sources.append(("마이크", sc.default_microphone()))
        if args.source in ("system", "both"):
            sys_dev = find_system_audio_device(args.system_device)
            if sys_dev is None:
                self.ui_q.put(
                    (
                        "error",
                        "시스템 소리 장치를 못 찾았습니다. Mac은 BlackHole 설치가 필요합니다 "
                        "(README 참고). 지금은 마이크만 씁니다.",
                    )
                )
                if not any(lbl == "마이크" for lbl, _ in sources):
                    sources.append(("마이크", sc.default_microphone()))
            else:
                sources.append(("시스템", sys_dev))

        queues = []
        for label, device in sources:
            q = queue.Queue(maxsize=24)  # 최대 12초치 버퍼만 유지 (무한 메모리 누적 방지)
            Recorder(device, q, self.stop_event, label).start()
            queues.append((label, q))
            self.ui_q.put(("status_add", f"{label}: {device.name}"))

        out_path = args.out

        if is_mlx_streaming:
            self._run_mlx_streaming_loop(model, queues, out_path, args)
        else:
            self._run_chunk_loop(model, queues, out_path, args)

    def _read_mixed_block(self, queues):
        mixed = None
        for _, q in queues:
            item = q.get()
            if isinstance(item, Exception):
                self.ui_q.put(("error", str(item)))
                return None
            mixed = item if mixed is None else mixed + item
        if len(queues) > 1:
            mixed = mixed / len(queues)
        return mixed

    def _commit_text(self, text: str, out_path: str):
        text = (text or "").strip()
        if not text:
            return
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {text}"
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.ui_q.put(("text", line))

    def _run_mlx_streaming_loop(self, engine, queues, out_path, args):
        """
        MLX 4-bit 실시간 스트리밍 루프:
        - 0.5초 블록마다 음성을 감지하고, 발화 중에는 1.0초마다 Prefix-Rollback 스트리밍으로
          중간 결과("partial")를 UI에 즉시 전송한다.
        - 발화 후 무음(1.0초 이상)이 이어지거나 최대 발화 길이(max_sec)에 도달하면
          문장을 확정("text")하고 파일에 저장한다.
        """
        import gc
        import mlx.core as mx

        max_utterance_sec = max(4.0, float(args.chunk) * 1.5)
        state = None
        silent_blocks = 0
        active_sec = 0.0
        step_counter = 0

        while not self.stop_event.is_set():
            block = self._read_mixed_block(queues)
            if block is None:
                return

            rms = float(np.sqrt(np.mean(block**2)))
            level_norm = min(1.0, rms / 0.08)
            step_counter += 1
            if step_counter % 2 == 0:
                active_mb = mx.get_active_memory() / (1024 * 1024)
                cache_mb = mx.get_cache_memory() / (1024 * 1024)
                self.ui_q.put(("telemetry", (round(level_norm, 3), round(active_mb + cache_mb, 1))))
            else:
                self.ui_q.put(("level", round(level_norm, 3)))

            is_silent = rms < args.silence

            if is_silent:
                if state is None:
                    continue
                silent_blocks += 1
                # 발화 중 잠깐 쉬는 호흡(0.5초)은 버퍼에 포함하고, 1.0초(2블록) 연속 무음이면 문장 확정
                if silent_blocks < 2:
                    engine.streaming_transcribe(block, state)
                    if state.text:
                        self.ui_q.put(("partial", state.text))
                    continue

                try:
                    engine.finish_streaming_transcribe(state)
                    self.ui_q.put(("partial", ""))
                    self._commit_text(state.text, out_path)
                except Exception as e:
                    self.ui_q.put(("error", f"스트리밍 마감 오류: {e}"))
                state = None
                silent_blocks = 0
                active_sec = 0.0
                mx.clear_cache()
                gc.collect()
                continue

            # 소리가 있는 블록
            silent_blocks = 0
            peak = float(np.max(np.abs(block)))
            if 0 < peak < 0.4:
                block = block * min(3.0, 0.4 / peak)

            if state is None:
                state = engine.init_streaming_state(
                    language="Korean",
                    chunk_size_sec=1.0,
                    unfixed_chunk_num=2,
                    unfixed_token_num=6,
                )

            prev_chunk_id = state.chunk_id
            try:
                engine.streaming_transcribe(block, state)
                active_sec += BLOCK_SEC
                if state.chunk_id != prev_chunk_id and state.text:
                    self.ui_q.put(("partial", state.text))
            except Exception as e:
                self.ui_q.put(("error", f"스트리밍 전사 오류: {e}"))
                state = None
                active_sec = 0.0
                mx.clear_cache()
                continue

            if active_sec >= max_utterance_sec:
                try:
                    engine.finish_streaming_transcribe(state)
                    self.ui_q.put(("partial", ""))
                    self._commit_text(state.text, out_path)
                except Exception as e:
                    self.ui_q.put(("error", f"스트리밍 구간 확정 오류: {e}"))
                state = None
                active_sec = 0.0
                mx.clear_cache()
                gc.collect()

    def _run_chunk_loop(self, model, queues, out_path, args):
        blocks_per_chunk = max(1, int(round(args.chunk / BLOCK_SEC)))
        while not self.stop_event.is_set():
            parts = []
            for _ in range(blocks_per_chunk):
                b = self._read_mixed_block(queues)
                if b is None:
                    return
                parts.append(b)
            mixed = np.concatenate(parts)

            rms = float(np.sqrt(np.mean(mixed**2)))
            if rms < args.silence:
                continue

            peak = float(np.max(np.abs(mixed)))
            if 0 < peak < 0.5:
                mixed = mixed * (0.5 / peak)

            try:
                text = transcribe_array(model, mixed)
            except Exception as e:
                self.ui_q.put(("error", f"전사 오류: {e}"))
                continue
            self._commit_text(text, out_path)


def main():
    import tkinter as tk
    from tkinter import scrolledtext

    p = argparse.ArgumentParser(description="실시간 한국어 전사 (Qwen3-ASR 1.7B ko)")
    p.add_argument("--model", default=None, help="모델 폴더 경로 또는 HF 모델 ID")
    p.add_argument("--source", choices=["both", "mic", "system"], default="both")
    p.add_argument("--chunk", type=float, default=6.0, help="최대 발화 구간 길이(초, 기본 6)")
    p.add_argument("--device", default="auto", help="auto / mlx / cuda:0 / mps / cpu")
    p.add_argument("--system-device", default=None, help="시스템 소리 장치 이름")
    p.add_argument("--silence", type=float, default=0.005, help="무음 임계값")
    p.add_argument("--topmost", action="store_true", help="창을 항상 위에 띄우기")
    p.add_argument("--out", default=None, help="저장할 txt 경로")
    p.add_argument("--list", action="store_true", help="오디오 장치 목록만 출력")
    args = p.parse_args()

    if args.list:
        list_devices()
        return 0

    if args.out is None:
        here = os.path.dirname(os.path.abspath(__file__))
        name = dt.datetime.now().strftime("transcript_%Y-%m-%d_%H%M.txt")
        args.out = os.path.join(here, name)

    root = tk.Tk()
    ui_q = queue.Queue()
    stop_event = threading.Event()
    root.title("실시간 전사 — qwen3_asr_1.7b_ko (MLX 4-bit)")
    root.geometry("720x480")
    root.attributes("-topmost", args.topmost)

    status = tk.Label(root, text="모델 로딩 중…", anchor="w", justify="left")
    status.pack(fill="x", padx=8, pady=(8, 0))
    box = scrolledtext.ScrolledText(
        root,
        wrap="word",
        font=("Malgun Gothic" if platform.system() == "Windows" else "Apple SD Gothic Neo", 15),
    )
    box.pack(fill="both", expand=True, padx=8, pady=8)
    partial_lbl = tk.Label(root, text="", anchor="w", fg="#10b981")
    partial_lbl.pack(fill="x", padx=8)
    tk.Label(root, text=f"저장 파일: {args.out}", anchor="w").pack(fill="x", padx=8, pady=(0, 8))

    Transcriber(args, ui_q, stop_event).start()

    def poll():
        try:
            while True:
                kind, msg = ui_q.get_nowait()
                if kind == "text":
                    box.insert("end", msg + "\n")
                    box.see("end")
                elif kind == "partial":
                    partial_lbl.config(text=f"🎙 {msg}" if msg else "")
                elif kind == "status":
                    status.config(text=msg)
                elif kind == "status_add":
                    status.config(text=status.cget("text") + "\n" + msg)
                elif kind == "error":
                    box.insert("end", "⚠ " + msg + "\n")
                    box.see("end")
        except queue.Empty:
            pass
        root.after(100, poll)

    def close():
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    poll()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
