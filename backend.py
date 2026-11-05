"""
전사 백엔드 (화면 없음) — GPUI 앱(app/)이 실행하고, stdout 을 한 줄씩 읽어 화면에 표시한다.

동작
- macOS (Apple Silicon): MLX 4-bit (yeowon/qwen3_asr_1.7b_ko-MLX-4bit) 실시간 1초 단위 스트리밍 전사 (live_transcribe.py 재사용).
- Windows / --use-vad: VAD(Silero)로 말이 끊기는 지점을 찾아 문장 단위로 전사하며, Windows 에서는 효율 모드(EcoQoS)를 끈다.

stdout 약속 (앱 ↔ 백엔드, UTF-8, 한 줄에 JSON 하나)
    {"type": "status",  "msg": "…"}
    {"type": "partial", "text": "실시간 스트리밍 인식 중…"}
    {"type": "text",    "text": "전사된 문장", "clock": "14:31:05",
     "start": 1730790665.2, "end": 1730790668.9, "infer_sec": 0.84}
    {"type": "error",   "msg": "…"}
로그·경고는 stderr 로만 나간다.
"""

import argparse
import datetime as dt
import json
import os
import platform
import queue
import sys
import threading
import time

SAMPLE_RATE = 16000
VAD_FRAME = 512  # Silero VAD 는 16kHz 에서 512 샘플(32ms) 단위


# ---------------------------------------------------------------- 출력
class Emitter:
    def __init__(self):
        # stdout 은 앱과 주고받는 통로라서, 라이브러리가 찍는 print 는 stderr 로 돌린다
        self.out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        self.lock = threading.Lock()
        self.broken = threading.Event()

    def __call__(self, **msg):
        try:
            with self.lock:
                self.out.write(json.dumps(msg, ensure_ascii=False) + "\n")
        except (BrokenPipeError, OSError):  # 앱이 사라짐
            self.broken.set()


def boost_process_windows():
    """창 없는 프로세스는 Windows 가 효율 모드로 돌려 CPU 가 느려진다 (측정: 토큰당 210ms → 110ms)."""
    if platform.system() != "Windows":
        return False
    import ctypes
    from ctypes import wintypes

    class POWER_THROTTLING(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                    ("StateMask", wintypes.ULONG)]

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    h = k.GetCurrentProcess()
    # ControlMask = 실행 속도·타이머 해상도 제어, StateMask = 0 → 둘 다 절전 안 함
    st = POWER_THROTTLING(1, 0x1 | 0x4, 0)
    ok = k.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st))  # 4 = ProcessPowerThrottling
    k.SetPriorityClass(h, 0x80)  # HIGH_PRIORITY_CLASS
    ctypes.WinDLL("winmm").timeBeginPeriod(1)
    return bool(ok)


# ---------------------------------------------------------------- 녹음 + VAD
class Segmenter(threading.Thread):
    """녹음 블록을 받아 VAD 로 말 구간을 찾고, 끝난 구간을 seg_q 에 넣는다."""

    def __init__(self, args, queues, seg_q, emit, stop):
        super().__init__(daemon=True)
        self.args, self.queues, self.seg_q, self.emit, self.stop = args, queues, seg_q, emit, stop

    def run(self):
        import numpy as np
        import torch
        from silero_vad import load_silero_vad

        vad = load_silero_vad(onnx=False)
        a = self.args
        fdur = VAD_FRAME / SAMPLE_RATE
        pre_frames = int(0.3 / fdur)
        end_frames = max(1, int(a.pause / fdur))     # 이만큼 조용하면 문장 끝
        max_frames = int(a.max_seg / fdur)
        min_frames = int(0.4 / fdur)                 # 말소리가 0.4초도 안 되면 버림 (기침, 클릭)
        search_frames = int(3.0 / fdur)              # 너무 길 때 마지막 3초 중 가장 조용한 곳에서 자른다

        leftover = np.zeros(0, dtype=np.float32)
        pre = []         # 말 시작 전 프레임 몇 개
        seg = []         # 지금 말하는 구간의 프레임들
        probs = []       # seg 각 프레임의 VAD 확률
        seg_start = 0.0
        silent = 0

        def voiced_count(ps):
            return sum(1 for x in ps if x >= a.vad)

        def emit_seg(frames, ps, start):
            if voiced_count(ps) >= min_frames:
                self.seg_q.put((np.concatenate(frames), start, start + len(frames) * fdur))

        while not self.stop.is_set():
            mixed = None
            for label, q in self.queues:
                item = q.get()
                if isinstance(item, Exception):
                    self.emit(type="error", msg=str(item))
                    return
                mixed = item if mixed is None else mixed[: len(item)] + item[: len(mixed)]
            if len(self.queues) > 1:
                mixed = mixed / len(self.queues)
            now = time.time()

            buf = np.concatenate([leftover, mixed])
            n = len(buf) // VAD_FRAME
            leftover = buf[n * VAD_FRAME:]
            t0 = now - len(buf) / SAMPLE_RATE

            for i in range(n):
                frame = buf[i * VAD_FRAME:(i + 1) * VAD_FRAME]
                t_frame = t0 + i * VAD_FRAME / SAMPLE_RATE
                with torch.no_grad():
                    prob = vad(torch.from_numpy(frame), SAMPLE_RATE).item()
                speech = prob >= a.vad

                if not seg:
                    if speech:
                        seg = pre + [frame]
                        probs = [0.0] * len(pre) + [prob]
                        seg_start = t_frame - len(pre) * fdur
                        silent = 0
                        pre = []
                    else:
                        pre = (pre + [frame])[-pre_frames:]
                    continue

                seg.append(frame)
                probs.append(prob)
                silent = 0 if speech else silent + 1

                if silent >= end_frames:
                    keep = len(seg) - silent + int(0.3 / fdur)
                    emit_seg(seg[:keep], probs[:keep], seg_start)
                    seg, probs, silent = [], [], 0
                elif len(seg) >= max_frames:
                    lo = max(min_frames, len(seg) - search_frames)
                    cut = min(range(lo, len(seg)), key=lambda j: probs[j])
                    emit_seg(seg[:cut], probs[:cut], seg_start)
                    seg_start += cut * fdur
                    seg, probs = seg[cut:], probs[cut:]


def feed_wavs(paths, q, stop, block_sec=0.2):
    """테스트용: wav 를 녹음처럼 block_sec 씩 실시간 속도로 큐에 넣는다. 파일 사이에 1초 무음."""
    import numpy as np
    import soundfile as sf

    block = int(SAMPLE_RATE * block_sec)
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    audio = []
    for p in paths:
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        if sr != SAMPLE_RATE:
            q.put(RuntimeError(f"{p}: {sr}Hz (16000Hz 만 지원)"))
            return
        audio += [x.mean(axis=1), silence]
    audio = np.concatenate([silence] + audio + [silence, silence])
    for i in range(0, len(audio) - block + 1, block):
        if stop.is_set():
            return
        q.put(audio[i:i + block])
        time.sleep(block_sec)
    while not stop.is_set():  # 끝나면 무음을 계속
        q.put(np.zeros(block, dtype=np.float32))
        time.sleep(block_sec)


# ---------------------------------------------------------------- 전사
def transcribe_loop(args, model, seg_q, emit, stop, out_path):
    context = args.context.strip()
    while not stop.is_set() and not emit.broken.is_set():
        try:
            audio, start, end = seg_q.get(timeout=0.2)
        except queue.Empty:
            continue

        backlog = seg_q.qsize()
        if backlog >= 3:
            emit(type="status", msg=f"전사가 밀리고 있어요: 대기 {backlog}개")

        peak = float(abs(audio).max())
        if 0 < peak < 0.5:  # 너무 작은 소리는 키운다 (클리핑 방지)
            audio = audio * (0.5 / peak)

        t = time.time()
        try:
            res = model.transcribe(audio=(audio, SAMPLE_RATE), context=context, language=args.language)
            text = (res[0].text or "").strip()
        except Exception as e:
            emit(type="error", msg=f"전사 오류: {type(e).__name__}: {e}")
            continue
        infer = time.time() - t
        if not text:
            continue

        clock = dt.datetime.fromtimestamp(start).strftime("%H:%M:%S")
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"[{clock}] {text}\n")
        emit(type="text", text=text, clock=clock, start=round(start, 3), end=round(end, 3),
             infer_sec=round(infer, 2))


def main():
    emit = Emitter()
    boosted = boost_process_windows()

    import live_transcribe as lt

    p = argparse.ArgumentParser(description="전사 백엔드 (GPUI 앱용, 화면 없음)")
    p.add_argument("--model", default=lt.MODEL_ID, help="모델 폴더 경로 또는 HF 모델 ID")
    p.add_argument("--source", choices=["both", "mic", "system"], default="both")
    p.add_argument("--chunk", type=float, default=6.0, help="최대 발화 구간 길이(초, 기본 6)")
    p.add_argument("--device", default="auto", help="auto / mlx / cuda:0 / mps / cpu")
    p.add_argument("--system-device", default=None, help="시스템 소리 장치 이름(일부만 써도 됨)")
    p.add_argument("--silence", type=float, default=0.005, help="이 값보다 조용하면 건너뜀")
    p.add_argument("--language", default="Korean", help="전사 언어 고정 (빈 값이면 자동 감지)")
    p.add_argument("--context", default="", help="자주 나오는 용어 (예: \"GPUI, 러스트, 백엔드\")")
    p.add_argument("--vad", type=float, default=0.5, help="말로 볼 VAD 확률 (0~1, 기본 0.5)")
    p.add_argument("--pause", type=float, default=0.6, help="이만큼(초) 조용하면 문장 끝 (기본 0.6)")
    p.add_argument("--max-seg", type=float, default=15.0, help="한 번에 전사할 최대 길이(초, 기본 15)")
    p.add_argument("--use-vad", action="store_true", help="Silero VAD 문장 단위 모드 강제 사용")
    p.add_argument("--out", default=None, help="저장할 txt 경로")
    p.add_argument("--input-wav", nargs="+", default=None,
                   help="테스트용: 녹음 대신 이 wav 들(16kHz mono)을 실시간 속도로 흘려 넣는다")
    args = p.parse_args()
    args.language = args.language or None

    if args.out is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.out = os.path.join(here, dt.datetime.now().strftime("transcript_%Y-%m-%d_%H%M.txt"))

    # macOS Apple Silicon 기본: MLX 4-bit 1초 단위 실시간 스트리밍 모드
    if platform.system() == "Darwin" and not args.input_wav and not args.use_vad:
        ui_q = queue.Queue()
        stop = threading.Event()
        emit(type="status", msg=f"모델 로딩 중… ({args.model})")
        emit(type="status", msg=f"저장 파일: {args.out}")
        worker = lt.Transcriber(args, ui_q, stop)
        worker.start()

        kinds = {
            "text": "text",
            "partial": "partial",
            "status": "status",
            "status_add": "status",
            "error": "error",
            "level": "level",
            "telemetry": "telemetry",
        }
        while not stop.is_set() and not emit.broken.is_set():
            try:
                kind, msg = ui_q.get(timeout=0.2)
            except queue.Empty:
                if not worker.is_alive():
                    return 1
                continue
            kind = kinds.get(kind, "status")
            if kind in ("text", "partial"):
                emit(type=kind, text=msg)
            elif kind == "level":
                emit(type="level", rms=float(msg))
            elif kind == "telemetry":
                rms_val, mem_mb = msg
                emit(type="telemetry", rms=float(rms_val), mem_mb=float(mem_mb))
            else:
                emit(type=kind, msg=msg)
        return 0

    emit(type="status", msg=f"모델 로딩 중… ({args.model})")
    try:
        model, dev = lt.load_model(args.device, args.model)
    except Exception as e:
        emit(type="error", msg=f"모델 로딩 실패: {e}")
        return 1

    # 녹음 블록을 짧게(0.2초) 해서 VAD 가 문장 끝을 빨리 알아채게 한다
    lt.BLOCK_SEC = 0.2
    stop = threading.Event()
    seg_q = queue.Queue()
    queues = []
    sources = []
    if args.input_wav:
        q = queue.Queue()
        threading.Thread(target=feed_wavs, args=(args.input_wav, q, stop), daemon=True).start()
        queues.append(("파일", q))
        emit(type="status", msg=f"테스트 입력: wav {len(args.input_wav)}개")
    elif args.source in ("mic", "both"):
        sources.append(("마이크", lt.sc.default_microphone()))
    if not args.input_wav and args.source in ("system", "both"):
        sys_dev = lt.find_system_audio_device(args.system_device)
        if sys_dev is None:
            emit(type="error", msg="시스템 소리 장치를 못 찾았습니다. 마이크만 씁니다.")
            if args.source == "system":
                sources.append(("마이크", lt.sc.default_microphone()))
        else:
            sources.append(("시스템", sys_dev))
    for label, device in sources:
        q = queue.Queue()
        lt.Recorder(device, q, stop, label).start()
        queues.append((label, q))
        emit(type="status", msg=f"{label}: {device.name}")

    Segmenter(args, queues, seg_q, emit, stop).start()
    emit(type="status", msg=(f"듣는 중… (device={dev}, VAD 문장 단위, 최대 {args.max_seg:g}초"
                             f"{', 효율 모드 끔' if boosted else ''}) · 저장: {args.out}"))
    try:
        transcribe_loop(args, model, seg_q, emit, stop, args.out)
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
