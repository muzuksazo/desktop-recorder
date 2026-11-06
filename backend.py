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
    """녹음 블록을 받아 VAD 로 말 구간을 찾고, 끝난 구간을 seg_q 에 넣는다.

    입력이 여러 개(마이크 + 시스템)면 섞어서 한 번만 전사한다 (스피커 소리가 마이크로 다시 잡혀
    같은 말이 두 번 나오는 걸 막기 위해). 대신 구간마다 어느 입력이 더 컸는지 보고 출처를 붙인다.
    """

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
        pre_en = []      # pre 각 프레임의 입력별 소리 크기 {라벨: rms}
        ens = []         # seg 각 프레임의 입력별 소리 크기
        labels = [lbl for lbl, _ in self.queues]
        single = labels[0] if len(labels) == 1 and labels[0] != "파일" else None
        level_t = 0.0
        seg_start = 0.0
        silent = 0

        def voiced_count(ps):
            return sum(1 for x in ps if x >= a.vad)

        def pick_source(es, ps):
            """말소리 프레임에서 소리가 더 컸던 입력 (입력이 하나면 그 입력, 파일 시험이면 None)"""
            if len(labels) == 1:
                return single
            tot = {lbl: 0.0 for lbl in labels}
            for e, p in zip(es, ps):
                if p >= a.vad:
                    for lbl, v in e.items():
                        tot[lbl] += v
            return max(tot, key=tot.get)

        def emit_seg(frames, ps, es, start):
            if voiced_count(ps) >= min_frames:
                self.seg_q.put((np.concatenate(frames), start, start + len(frames) * fdur,
                                pick_source(es, ps)))

        while not self.stop.is_set():
            mixed = None
            block_en = {}
            for label, q in self.queues:
                item = q.get()
                if isinstance(item, Exception):
                    self.emit(type="error", msg=str(item))
                    return
                block_en[label] = float(np.sqrt(np.mean(item ** 2))) if len(item) else 0.0
                mixed = item if mixed is None else mixed[: len(item)] + item[: len(mixed)]
            if len(self.queues) > 1:
                mixed = mixed / len(self.queues)
            now = time.time()
            if now - level_t >= 0.1:  # 화면의 소리 크기 막대용 (0~1)
                level_t = now
                self.emit(type="level", rms=round(min(1.0, max(block_en.values(), default=0.0) / 0.08), 3))

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
                        ens = pre_en + [block_en]
                        seg_start = t_frame - len(pre) * fdur
                        silent = 0
                        pre, pre_en = [], []
                    else:
                        pre = (pre + [frame])[-pre_frames:]
                        pre_en = (pre_en + [block_en])[-pre_frames:]
                    continue

                seg.append(frame)
                probs.append(prob)
                ens.append(block_en)
                silent = 0 if speech else silent + 1

                if silent >= end_frames:
                    keep = len(seg) - silent + int(0.3 / fdur)
                    emit_seg(seg[:keep], probs[:keep], ens[:keep], seg_start)
                    seg, probs, ens, silent = [], [], [], 0
                elif len(seg) >= max_frames:
                    lo = max(min_frames, len(seg) - search_frames)
                    cut = min(range(lo, len(seg)), key=lambda j: probs[j])
                    emit_seg(seg[:cut], probs[:cut], ens[:cut], seg_start)
                    seg_start += cut * fdur
                    seg, probs, ens = seg[cut:], probs[cut:], ens[cut:]


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
            audio, start, end, source = seg_q.get(timeout=0.2)
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
        # 출처가 있으면 Mac 경로와 같은 모양으로 붙인다: "[시스템] 문장" (앱은 이 머리말로 출처를 표시)
        labeled = f"[{source}] {text}" if source else text
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"[{clock}] {labeled}\n")
        emit(type="text", text=labeled, clock=clock, start=round(start, 3), end=round(end, 3),
             infer_sec=round(infer, 2), source=source)


# ---------------------------------------------------------------- 모델 목록·받기·불러오기
# 앱에서 고를 수 있는 음성 인식 모델. 모두 공개라서 로그인 없이 Hugging Face 에서 받는다.
# 받은 모델은 ~/Library/Application Support/Talkit/models/<저장소 이름> 에 둔다 (앱의 updates.rs 와 같은 규칙).
MODEL_REPOS = {
    "public": "yeowon/qwen3_asr_1.7b_ko-MLX-4bit",
    "rung-omega": "yeowon/qwen3_asr_1.7b_ko-RungOmega-MLX",
    "rung-omega-lite": "yeowon/qwen3_asr_1.7b_ko-RungOmega-MLX-lite",
    "fa": "yeowon/qwen3_forced_aligner_0.6b-RungOmega-MLX",  # 문장 맞추기 (Forced Aligner)
}
# 화면에서 고르는 항목: (id, 이름, 음성 인식 모델, 문장 맞추기 사용)
CHOICES = [
    ("public", "기본 모델", "public", False),
    ("rung-omega", "Rung Ω", "rung-omega", False),
    ("rung-omega-fa", "Rung Ω + 문장 맞추기", "rung-omega", True),
    ("rung-omega-lite", "Rung Ω 가벼운판", "rung-omega-lite", False),
]
# 개발용 Mac: 받지 않았어도 예전 작업 폴더의 같은 가중치를 쓴다 (외부 배포판은 보지 않음)
_DEV_ARTIFACTS = {
    "rung-omega": "qwen3_asr_1_7b_ko_rung_omega_g128_mlx",
    "rung-omega-lite": "qwen3_asr_1_7b_ko_rung_omega_mac_mlx",
    "fa": "qwen3_forced_aligner_0_6b_rung_omega_mac_mlx",
}


def is_public_edition():
    """외부 배포판(DMG)이면 True. 패키징 스크립트가 앱 안에 edition.txt(public)를 넣는다."""
    from pathlib import Path

    f = Path(__file__).resolve().parent / "edition.txt"
    try:
        return f.read_text(encoding="utf-8").strip() == "public"
    except OSError:
        return False


def _dev_artifacts_dir():
    from pathlib import Path

    if is_public_edition():
        return None
    for root in (
        Path.home() / "Library" / "Application Support" / "Talkit" / "engines" / "rung-omega",
        Path.home() / "qwen3-asr-ko" / "private_rung_omega_mac",
    ):
        if (root / "artifacts").is_dir():
            return root / "artifacts"
    return None


def model_dir(mid):
    """받아 둔 모델 폴더 (없으면 None)"""
    try:
        import mlx_quantized_asr as mqa
    except ImportError:  # Windows 등 MLX 가 없는 PC: 이 목록의 모델(MLX 형식)은 쓸 수 없다
        return None

    repo = MODEL_REPOS[mid]
    target = mqa.talkit_models_dir() / repo.split("/")[-1]
    if mqa.model_is_complete(target):
        return target
    if mid == "public":
        legacy = mqa.Path.home() / "qwen3-asr-ko" / "qwen3_asr_1.7b_ko_mlx_4bit"
        return legacy if (legacy / "model.safetensors").exists() else None
    art = _dev_artifacts_dir()
    if art is not None and (art / _DEV_ARTIFACTS[mid] / "model.safetensors").exists():
        return art / _DEV_ARTIFACTS[mid]
    return None


def list_models():
    """고를 수 있는 모델 전부와 이 Mac 에 받아 뒀는지"""
    items = []
    for cid, label, asr, fa in CHOICES:
        installed = model_dir(asr) is not None and (not fa or model_dir("fa") is not None)
        items.append({"id": cid, "label": label, "installed": installed})
    return items


def _install_rung_engine(lt, choice, emit):
    """고른 Rung Ω 구성으로 모델 로더를 바꾼다. 모델이 없거나 실패하면 False (기본 모델로 계속)."""
    import sys

    entry = next((c for c in CHOICES if c[0] == choice), None)
    if entry is None or entry[2] == "public":
        return False
    _, label, asr, use_fa = entry
    asr_dir = model_dir(asr)
    fa_dir = model_dir("fa") if use_fa else None
    if asr_dir is None or (use_fa and fa_dir is None):
        emit(type="error", msg=f"'{label}' 모델을 아직 받지 않았어요. 모델 화면에서 받은 뒤 다시 골라 주세요. 지금은 기본 모델로 전사합니다.")
        return False
    try:
        import mlx.core as mx
        import rung_runtime

        def _load_rung(device_arg, model_id, mlx_4bit=True):
            mx.set_cache_limit(128 * 1024 * 1024)
            engine = rung_runtime.MacTheoremFlowOmegaASREngine.from_omega_dir(asr_dir, fa_dir=fa_dir)
            # 이 이름은 live_transcribe 가 'mlx' 로 시작하는지로 실시간 스트리밍 방식을 고르는 데 쓴다.
            # 화면용 이름(label)을 넣으면 옛 덩어리 방식으로 빠져 '.transcribe 없음' 오류가 난다(2026-11-06).
            return engine, f"mlx-{choice}"

        lt.load_model = _load_rung
        sys.stderr.write(f"[Talkit] {label} 사용: {asr_dir}{' + ' + str(fa_dir) if fa_dir else ''}\n")
        return True
    except Exception as e:  # 깨진 모델이어도 앱은 기본 모델로 계속 동작
        emit(type="error", msg=f"'{label}' 을 불러오지 못해 기본 모델로 전사합니다: {e}")
        return False


def download_model_only(emit, mid):
    """앱의 '모델' 화면용: 모델 하나만 내려받고 끝낸다 (녹음·전사는 하지 않음).
    보내는 줄: download(진행) → download_done(끝) 또는 error(실패)."""
    import mlx_quantized_asr as mqa

    if mid not in MODEL_REPOS:
        emit(type="error", msg=f"모르는 모델: {mid}")
        return 1
    mqa.PROGRESS_HOOK = None
    mqa.DOWNLOAD_HOOK = lambda got, total: emit(type="download", got=int(got), total=int(total) if total else None)
    try:
        path = mqa.resolve_model_dir(MODEL_REPOS[mid])
    except Exception as e:  # 인터넷 끊김·용량 부족 등: 그대로 화면에 보여 준다
        emit(type="error", msg=f"{e}")
        return 1
    emit(type="download_done", path=str(path))
    return 0


def main():
    emit = Emitter()
    if "--download-model" in sys.argv[1:]:
        i = sys.argv.index("--download-model")
        nxt = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
        return download_model_only(emit, nxt if nxt and not nxt.startswith("-") else "public")
    boosted = boost_process_windows()

    import live_transcribe as lt

    p = argparse.ArgumentParser(description="전사 백엔드 (GPUI 앱용, 화면 없음)")
    p.add_argument("--model", default=lt.MODEL_ID, help="모델 폴더 경로 또는 HF 모델 ID")
    p.add_argument("--source", choices=["both", "mic", "system"], default="both")
    p.add_argument("--model-choice", default="", help="쓸 모델 id (public / rung-omega / rung-omega-fa / rung-omega-lite)")
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
    p.add_argument("--quant", default=None,
                   help="Windows/VAD 경로 양자화: auto / none / int8 / nf4 / cpu-int8 (model_tool.load_asr)")
    p.add_argument("--out", default=None, help="저장할 txt 경로")
    p.add_argument("--input-wav", nargs="+", default=None,
                   help="테스트용: 녹음 대신 이 wav 들(16kHz mono)을 실시간 속도로 흘려 넣는다")
    args, _ = p.parse_known_args()

    # 쓸 수 있는 모델을 앱에 알려 주고, 고른 모델을 준비한다
    models = list_models()
    ids = [m["id"] for m in models]
    choice = args.model_choice if args.model_choice in ids else "public"
    if choice != "public" and not _install_rung_engine(lt, choice, emit):
        choice = "public"
    emit(type="models", items=models, current=choice)
    args.language = args.language or None

    if args.out is None:
        here = os.path.dirname(os.path.abspath(__file__))
        if ".app/Contents/Resources" in here or not os.access(here, os.W_OK):
            out_dir = os.path.expanduser("~/Documents/Talkit")
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = here
        args.out = os.path.join(out_dir, dt.datetime.now().strftime("transcript_%Y-%m-%d_%H%M.txt"))

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
        if args.quant and platform.system() != "Darwin":
            # Windows 판: 양자화(8-bit/4-bit/CPU 8-bit)를 고를 수 있는 불러오기
            from model_tool import load_asr
            model, dev = load_asr(args.model, args.quant, args.device)
        else:
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
