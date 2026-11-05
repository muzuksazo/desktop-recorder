"""
실시간 한국어 전사 (마이크 + 시스템 소리) — TeamUNIVA/qwen3_asr_1.7b_ko

동작 방식
- 마이크와 시스템 소리(스피커 출력)를 동시에 녹음해 섞는다.
- CHUNK_SEC 초마다 잘라서 모델로 전사하고, 창에 이어 붙인다.
- 전사될 때마다 transcript_YYYY-MM-DD_HHMM.txt 에 한 줄씩 바로 저장한다.

실행 예
    python live_transcribe.py                 # 마이크 + 시스템 소리
    python live_transcribe.py --source mic    # 마이크만
    python live_transcribe.py --chunk 3       # 3초 단위로 전사
    python live_transcribe.py --list          # 오디오 장치 목록 보기
"""

import argparse
import datetime as dt
import os
import platform
import queue
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import scrolledtext

import numpy as np
import soundcard as sc
import soundfile as sf

MODEL_ID = "TeamUNIVA/qwen3_asr_1.7b_ko"
SAMPLE_RATE = 16000
BLOCK_SEC = 0.5  # 녹음 스레드가 한 번에 읽는 길이


# ---------------------------------------------------------------- 오디오 장치
def find_system_audio_device(name_hint=None):
    """시스템 소리(루프백) 녹음 장치를 찾는다.

    Windows: 기본 스피커의 루프백(WASAPI loopback)을 쓴다. 추가 설치 없음.
    Mac: 기본 기능으로는 안 되므로 BlackHole 같은 가상 장치가 필요하다.
    """
    if name_hint:
        return sc.get_microphone(name_hint, include_loopback=True)

    if platform.system() == "Windows":
        return sc.get_microphone(sc.default_speaker().name, include_loopback=True)

    # Mac / Linux: 이름에 BlackHole 이 들어간 입력 장치를 찾는다
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
    """장치 하나에서 계속 녹음해서 out_q 에 0.5초 블록을 넣는다."""

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
                    self.out_q.put(data[:, 0].astype(np.float32))
        except Exception as e:  # 장치 오류는 창에 표시
            self.out_q.put(RuntimeError(f"[{self.label}] 녹음 오류: {e}"))


# ---------------------------------------------------------------- 모델
def pick_device(requested):
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


def load_model(device_arg, model_id=MODEL_ID):
    from qwen_asr import Qwen3ASRModel

    dev, dtype = pick_device(device_arg)
    print(f"모델 로딩 중... (device={dev}, dtype={dtype})")
    model = Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=dev,
        max_inference_batch_size=1,
        max_new_tokens=256,
    )
    print("모델 로딩 완료")
    return model, dev


def transcribe_array(model, audio):
    """공식 예제는 파일 경로 입력만 안내하므로, 임시 wav 로 저장해서 넘긴다."""
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
            model, dev = load_model(args.device, getattr(args, "model", None) or MODEL_ID)
        except Exception as e:
            self.ui_q.put(("error", f"모델 로딩 실패: {e}"))
            return
        self.ui_q.put(("status", f"듣는 중… (device={dev}, {args.chunk}초 단위)"))

        # 장치 준비
        sources = []
        if args.source in ("mic", "both"):
            sources.append(("마이크", sc.default_microphone()))
        if args.source in ("system", "both"):
            sys_dev = find_system_audio_device(args.system_device)
            if sys_dev is None:
                self.ui_q.put(("error",
                               "시스템 소리 장치를 못 찾았습니다. Mac은 BlackHole 설치가 필요합니다 "
                               "(README 참고). 지금은 마이크만 씁니다."))
                if not any(lbl == "마이크" for lbl, _ in sources):
                    sources.append(("마이크", sc.default_microphone()))
            else:
                sources.append(("시스템", sys_dev))

        queues = []
        for label, device in sources:
            q = queue.Queue()
            Recorder(device, q, self.stop_event, label).start()
            queues.append((label, q))
            self.ui_q.put(("status_add", f"{label}: {device.name}"))

        blocks_per_chunk = max(1, int(round(args.chunk / BLOCK_SEC)))
        out_path = args.out
        while not self.stop_event.is_set():
            # 각 장치에서 chunk 길이만큼 모아서 섞는다
            mixed = None
            for label, q in queues:
                parts = []
                for _ in range(blocks_per_chunk):
                    item = q.get()
                    if isinstance(item, Exception):
                        self.ui_q.put(("error", str(item)))
                        return
                    parts.append(item)
                audio = np.concatenate(parts)
                mixed = audio if mixed is None else mixed + audio

            if len(queues) > 1:
                mixed = mixed / len(queues)

            # 소리가 거의 없으면 건너뛴다
            rms = float(np.sqrt(np.mean(mixed ** 2)))
            if rms < args.silence:
                continue

            # 너무 작은 소리는 키운다 (클리핑 방지)
            peak = float(np.max(np.abs(mixed)))
            if 0 < peak < 0.5:
                mixed = mixed * (0.5 / peak)

            try:
                text = transcribe_array(model, mixed)
            except Exception as e:
                self.ui_q.put(("error", f"전사 오류: {e}"))
                continue
            if not text:
                continue

            stamp = dt.datetime.now().strftime("%H:%M:%S")
            line = f"[{stamp}] {text}"
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self.ui_q.put(("text", line))


# ---------------------------------------------------------------- 창
class App:
    def __init__(self, root, args):
        self.root = root
        self.ui_q = queue.Queue()
        self.stop_event = threading.Event()

        root.title("실시간 전사 — qwen3_asr_1.7b_ko")
        root.geometry("720x480")
        root.attributes("-topmost", args.topmost)

        self.status = tk.Label(root, text="모델 로딩 중… (처음에는 다운로드 때문에 몇 분 걸립니다)",
                               anchor="w", justify="left")
        self.status.pack(fill="x", padx=8, pady=(8, 0))

        self.box = scrolledtext.ScrolledText(root, wrap="word", font=("Malgun Gothic"
                                             if platform.system() == "Windows" else "Apple SD Gothic Neo", 15))
        self.box.pack(fill="both", expand=True, padx=8, pady=8)

        tk.Label(root, text=f"저장 파일: {args.out}", anchor="w").pack(fill="x", padx=8, pady=(0, 8))

        Transcriber(args, self.ui_q, self.stop_event).start()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.poll()

    def poll(self):
        try:
            while True:
                kind, msg = self.ui_q.get_nowait()
                if kind == "text":
                    self.box.insert("end", msg + "\n")
                    self.box.see("end")
                elif kind == "status":
                    self.status.config(text=msg)
                elif kind == "status_add":
                    self.status.config(text=self.status.cget("text") + "\n" + msg)
                elif kind == "error":
                    self.box.insert("end", "⚠ " + msg + "\n")
                    self.box.see("end")
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def close(self):
        self.stop_event.set()
        self.root.destroy()


def main():
    p = argparse.ArgumentParser(description="실시간 한국어 전사 (Qwen3-ASR 1.7B ko)")
    p.add_argument("--source", choices=["both", "mic", "system"], default="both",
                   help="녹음할 소리 (기본: both)")
    p.add_argument("--chunk", type=float, default=4.0, help="몇 초 단위로 전사할지 (기본 4)")
    p.add_argument("--device", default="auto", help="auto / cuda:0 / mps / cpu")
    p.add_argument("--system-device", default=None, help="시스템 소리 장치 이름(일부만 써도 됨)")
    p.add_argument("--silence", type=float, default=0.005, help="이 값보다 조용하면 건너뜀")
    p.add_argument("--topmost", action="store_true", help="창을 항상 위에 띄우기")
    p.add_argument("--out", default=None, help="저장할 txt 경로")
    p.add_argument("--list", action="store_true", help="오디오 장치 목록만 출력")
    args = p.parse_args()

    if args.list:
        list_devices()
        return

    if args.out is None:
        here = os.path.dirname(os.path.abspath(__file__))
        name = dt.datetime.now().strftime("transcript_%Y-%m-%d_%H%M.txt")
        args.out = os.path.join(here, name)

    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
