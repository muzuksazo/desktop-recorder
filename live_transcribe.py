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
import time

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
BLOCK_SEC = 0.25  # 250ms 고해상도 블록 (어두 자음 보존 및 호흡 골짜기 정밀 감지)

DEFAULT_KO_DOMAIN_CONTEXT = (
    "한국어 실시간 음성 전사. GitHub, Claude, HuggingFace, OpenAI, Three.js, WebGL, "
    "GPUI, Rust, Python, MLX, 프론트엔드, 백엔드, 라이선스, 레퍼런스, UI, PiP, TTS, ASR"
)


def _speech_band_rms(samples: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    인간 음성 기본주파수 및 한국어 치찰음/마찰음(ㅅ, ㅆ, ㅈ, ㅊ, ㅎ) 대역(150Hz ~ 7,500Hz) 에너지를 추출한다.
    극저주파 험(<120Hz)만 배제하고 어말 자음/마찰음까지 온전히 감지해 단어 중간 끊김을 방지한다.
    """
    if samples is None or samples.size == 0:
        return 0.0
    spec = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sr)
    mask = (freqs >= 150.0) & (freqs <= 7500.0)
    if not np.any(mask):
        return float(np.sqrt(np.mean(samples**2)))
    band_power = np.sum(np.abs(spec[mask]) ** 2) * 2.0 / (samples.size**2)
    return float(np.sqrt(max(0.0, band_power)))


def _find_valley_split_sample(
    audio_accum: np.ndarray,
    min_sec: float = 5.5,
    max_sec: float = 10.5,
    sr: int = SAMPLE_RATE,
) -> int:
    """
    Official `qwen_asr.inference.utils.split_audio_into_chunks` exact boundary finder:
    Finds the 100ms window with minimum L1 energy in [min_sec, max_sec] via `np.convolve`
    and snaps to the exact minimum-amplitude sample inside that window (zero-crossing snap).
    """
    total = int(audio_accum.shape[0])
    left = max(int(min_sec * sr), int(total * 0.50))
    right = min(int(max_sec * sr), total - int(0.30 * sr))
    win = max(4, int(0.10 * sr))  # 100ms window
    if right - left <= win:
        return total

    seg = audio_accum[left:right]
    seg_abs = np.abs(seg)
    window_sums = np.convolve(seg_abs, np.ones(win, dtype=np.float32), mode="valid")
    min_pos = int(np.argmin(window_sums))
    local = seg_abs[min_pos : min_pos + win]
    inner = int(np.argmin(local))
    boundary = left + min_pos + inner
    return int(max(left + 1, min(boundary, total)))


def _stitch_overlap_text(prev_text: str, new_text: str) -> str:
    """
    Removes duplicate boundary words or partial-syllable overlaps when consecutive segments
    share an overlapping suffix/prefix (e.g. "하고" + "하고 GEO", "목표로" + "표를 했다면").
    """
    prev_text = (prev_text or "").strip()
    new_text = (new_text or "").strip()
    if not prev_text or not new_text:
        return new_text

    prev_words = prev_text.split()
    new_words = new_text.split()
    max_k = min(len(prev_words), len(new_words), 10)
    for k in range(max_k, 0, -1):
        p_tail = "".join(prev_words[-k:]).rstrip(".?!,~")
        n_head = "".join(new_words[:k]).rstrip(".?!,~")
        min_len = 2 if k == 1 else 3
        if len(p_tail) >= min_len and p_tail == n_head:
            return " ".join(new_words[k:]).strip()

    # Partial word suffix overlap (e.g. prev ends with "목표로", new starts with "표를")
    p_last = prev_words[-1].rstrip(".?!,~")
    n_first = new_words[0].rstrip(".?!,~")
    if len(p_last) >= 3 and len(n_first) >= 2 and p_last != n_first and p_last.endswith(n_first):
        return " ".join(new_words[1:]).strip()

    return new_text


def _suppress_speaker_echo_from_mic(
    mic_block: np.ndarray, sys_history: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    ScreenCaptureKit 디지털 시스템 원음(sys_history)을 레퍼런스로 사용하는
    3단계 음향 에코 제거기(Reference-Based Acoustic Echo Cancellation, AEC):
    1) FFT 교차상관(Cross-Correlation) 및 10ms 에너지 포락선(Envelope) 정렬로 스피커-마이크 지연(Delay) 추정
    2) 주파수 영역(STFT) 적응형 위너 스펙트럼 감산(Wiener Spectral Echo Subtraction)으로 스피커 잔향 제거
    3) 파형 상관계수 및 포락선 일관성 점수(echo_coherence)를 반환해 더블토크(사용자 실제 발화) 여부 판별
    """
    n = mic_block.shape[0]
    m = sys_history.shape[0]
    if n < 320 or m < n:
        return mic_block, 0.0

    sys_rms = float(np.sqrt(np.mean(sys_history**2)))
    mic_rms = float(np.sqrt(np.mean(mic_block**2)))
    if sys_rms < 0.0025 or mic_rms < 0.0005:
        return mic_block, 0.0

    # 1. FFT 교차상관으로 최적 지연 오프셋 탐색
    fft_len = 1 << int(np.ceil(np.log2(m + n)))
    X = np.fft.rfft(sys_history, n=fft_len)
    Y = np.fft.rfft(mic_block, n=fft_len)
    corr = np.fft.irfft(X * np.conj(Y), n=fft_len)
    valid_len = m - n + 1
    valid_corr = corr[:valid_len]
    best_offset = int(np.argmax(np.abs(valid_corr)))
    sys_aligned = sys_history[best_offset : best_offset + n]

    sys_al_norm = float(np.linalg.norm(sys_aligned))
    mic_norm = float(np.linalg.norm(mic_block))
    if sys_al_norm < 1e-6 or mic_norm < 1e-6:
        return mic_block, 0.0

    dot_val = float(np.dot(mic_block, sys_aligned))
    rho_wave = abs(dot_val) / (mic_norm * sys_al_norm + 1e-8)

    # 10ms(160샘플) 서브프레임 포락선(Envelope) 상관계수 계산 (스피커 위상 왜곡/방 잔향에도 강인함)
    hop = 160
    n_frames = n // hop
    if n_frames >= 4:
        mic_env = np.sqrt(
            np.mean(mic_block[: n_frames * hop].reshape(n_frames, hop) ** 2, axis=1)
        )
        sys_env = np.sqrt(
            np.mean(sys_aligned[: n_frames * hop].reshape(n_frames, hop) ** 2, axis=1)
        )
        m_centered = mic_env - np.mean(mic_env)
        s_centered = sys_env - np.mean(sys_env)
        denom = float(np.linalg.norm(m_centered) * np.linalg.norm(s_centered))
        rho_env = float(np.dot(m_centered, s_centered) / (denom + 1e-8)) if denom > 1e-7 else 0.0
    else:
        rho_env = 0.0

    echo_coherence = max(rho_wave, max(0.0, rho_env))

    # 2. 시간 영역 최소제곱 선형 에코 감산 + 주파수 영역(STFT) 스펙트럼 에코 억제
    alpha = dot_val / (sys_al_norm * sys_al_norm + 1e-8)
    td_cleaned = mic_block - np.clip(alpha, -1.5, 1.5) * sys_aligned

    win_len = 320
    stft_hop = 160
    if n >= win_len:
        from numpy.lib.stride_tricks import sliding_window_view

        win = np.hanning(win_len).astype(np.float32)
        mic_frames = sliding_window_view(td_cleaned, win_len)[::stft_hop] * win
        sys_frames = sliding_window_view(sys_aligned, win_len)[::stft_hop] * win
        M_spec = np.fft.rfft(mic_frames, axis=-1)
        S_spec = np.fft.rfft(sys_frames, axis=-1)
        M_mag = np.abs(M_spec)
        S_mag = np.abs(S_spec)

        # 주파수 빈별 평균 에코 전달비 H(f) 추정 후 과감산(Over-subtraction factor = 1.45)
        H_f = np.sum(M_mag * S_mag, axis=0, keepdims=True) / (
            np.sum(S_mag * S_mag, axis=0, keepdims=True) + 1e-8
        )
        gain = np.clip(1.0 - 1.45 * (H_f * S_mag) / (M_mag + 1e-8), 0.04, 1.0)
        cleaned_frames = np.fft.irfft(M_spec * gain, n=win_len, axis=-1).astype(np.float32)

        out = np.zeros((n,), dtype=np.float32)
        w_sum = np.zeros((n,), dtype=np.float32)
        for idx_f in range(cleaned_frames.shape[0]):
            start = idx_f * stft_hop
            out[start : start + win_len] += cleaned_frames[idx_f] * win
            w_sum[start : start + win_len] += win * win
        mask = w_sum > 1e-5
        out[mask] /= w_sum[mask]
        tail_start = (cleaned_frames.shape[0] - 1) * stft_hop + win_len
        if tail_start < n:
            out[tail_start:] = td_cleaned[tail_start:]
        return out, echo_coherence

    return td_cleaned.astype(np.float32), echo_coherence


def _normalize_ko_for_echo(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isalnum())


def _is_text_echo_of_system(mic_text: str, sys_texts: list[str]) -> bool:
    """
    마이크 채널에서 전사된 텍스트가 시스템 오디오에서 방금 출력된 문장(또는 진행 중인 문장)의
    스피커 역유입(Bleed) 복사본인지 2-gram 및 부분문자열 유사도로 판별한다.
    """
    m_norm = _normalize_ko_for_echo(mic_text)
    if not m_norm:
        return False
    for st in sys_texts:
        s_norm = _normalize_ko_for_echo(st)
        if not s_norm:
            continue
        if m_norm in s_norm or (len(s_norm) >= 4 and s_norm in m_norm):
            return True
        if len(m_norm) >= 3 and len(s_norm) >= 3:
            m_bigrams = {m_norm[i : i + 2] for i in range(len(m_norm) - 1)}
            s_bigrams = {s_norm[i : i + 2] for i in range(len(s_norm) - 1)}
            if m_bigrams:
                overlap = len(m_bigrams & s_bigrams) / float(len(m_bigrams))
                if overlap >= 0.35:
                    return True
    return False


# MLX 를 못 불러온 진짜 이유 (torch 쪽으로 넘어가 'No module named torch' 로 가려지지 않게 보관)
_MLX_ERROR = None


def _has_mlx() -> bool:
    global _MLX_ERROR
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
        import qwen3_asr_mlx  # noqa: F401
        return True
    except Exception as e:
        _MLX_ERROR = e
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


def find_macos_audio_tap_binary():
    """macOS 내장 ScreenCaptureKit 기반 시스템 오디오 캡처 헬퍼 바이너리 경로를 찾는다."""
    if platform.system() != "Darwin":
        return None
    from pathlib import Path
    import subprocess

    base_dir = Path(__file__).resolve().parent
    cache_bin = Path.home() / ".cache" / "talkit" / "macos_audio_tap"
    candidates = [
        base_dir / "macos_audio_tap",
        cache_bin,
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)

    swift_src = base_dir / "macos_audio_tap.swift"
    target_bin = base_dir / "macos_audio_tap" if os.access(base_dir, os.W_OK) else cache_bin
    if swift_src.is_file():
        try:
            target_bin.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["swiftc", "-O", "-parse-as-library", str(swift_src), "-o", str(target_bin)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
            if target_bin.is_file() and os.access(target_bin, os.X_OK):
                return str(target_bin)
        except Exception:
            pass
    return None


class ScreenCaptureKitRecorder(threading.Thread):
    """macOS ScreenCaptureKit 헬퍼(macos_audio_tap)로부터 16kHz mono float32 시스템 오디오를 읽는다."""

    def __init__(self, tap_bin, out_q, stop_event, label="시스템"):
        super().__init__(daemon=True)
        self.tap_bin = tap_bin
        self.out_q = out_q
        self.stop_event = stop_event
        self.label = label
        self.name = "macOS ScreenCaptureKit (시스템 오디오)"

    def run(self):
        import subprocess

        frames = int(SAMPLE_RATE * BLOCK_SEC)
        bytes_per_block = frames * 4  # float32 = 4 bytes
        proc = None
        try:
            proc = subprocess.Popen(
                [self.tap_bin],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=bytes_per_block * 2,
            )
            while not self.stop_event.is_set():
                raw = proc.stdout.read(bytes_per_block) if proc.stdout else b""
                if not raw or len(raw) < bytes_per_block:
                    if self.stop_event.is_set():
                        break
                    err_msg = ""
                    if proc.stderr:
                        try:
                            err_msg = proc.stderr.read().decode("utf-8", errors="replace").strip()
                        except Exception:
                            pass
                    raise RuntimeError(err_msg or "ScreenCaptureKit 오디오 스트림이 종료되었습니다.")

                block = np.frombuffer(raw, dtype=np.float32).copy()
                try:
                    self.out_q.put_nowait(block)
                except queue.Full:
                    try:
                        self.out_q.get_nowait()
                    except queue.Empty:
                        pass
                    self.out_q.put_nowait(block)
        except Exception as e:
            if not self.stop_event.is_set():
                try:
                    self.out_q.put_nowait(RuntimeError(f"[{self.label}] 녹음 오류: {e}"))
                except queue.Full:
                    pass
        finally:
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass


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

    try:
        import torch
    except ImportError:
        # Mac 앱(DMG)에는 torch 가 없다. Apple Silicon 에서 여기까지 왔다면 MLX 가 안 열린 것이다.
        if _MLX_ERROR is not None:
            raise RuntimeError(
                f"음성 인식 엔진(MLX)을 열지 못했어요. macOS 14.4 이상이 필요하고, 이미 그 이상이면 최신 Talkit 을 다시 설치해 주세요. "
                f"(macOS {platform.mac_ver()[0]}, 원인: {_MLX_ERROR})"
            ) from None
        raise RuntimeError(
            "이 컴퓨터에서 쓸 음성 인식 엔진이 없어요. Talkit Mac 앱은 Apple Silicon(M1 이상) Mac 전용입니다."
        ) from None

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
        # 첫 실행에 모델을 내려받는 동안 진행률(%)을 앱 상태 줄에 보여 준다
        try:
            import mlx_quantized_asr as _mqa

            _mqa.PROGRESS_HOOK = lambda msg: self.ui_q.put(("status", msg))
        except Exception:
            pass
        try:
            model, dev = load_model(args.device, getattr(args, "model", None))
        except Exception as e:
            self.ui_q.put(("error", f"모델 로딩 실패: {e}"))
            return

        is_mlx_streaming = dev.startswith("mlx")
        mode_desc = "실시간 1초 스트리밍" if is_mlx_streaming else f"{args.chunk}초 단위"
        self.ui_q.put(("status", f"듣는 중… (engine={dev}, {mode_desc})"))

        # 장치 준비
        queues = []
        if args.source in ("mic", "both"):
            mic_dev = sc.default_microphone()
            q = queue.Queue(maxsize=64)
            Recorder(mic_dev, q, self.stop_event, "마이크").start()
            queues.append(["마이크", q])
            self.ui_q.put(("status_add", f"마이크: {mic_dev.name}"))

        if args.source in ("system", "both"):
            sys_dev = find_system_audio_device(args.system_device) if args.system_device else None
            tap_bin = find_macos_audio_tap_binary() if sys_dev is None else None
            if tap_bin is not None:
                q = queue.Queue(maxsize=64)
                ScreenCaptureKitRecorder(tap_bin, q, self.stop_event, "시스템").start()
                queues.append(["시스템", q])
                self.ui_q.put(("status_add", "시스템: macOS 시스템 오디오 탭"))
            else:
                if sys_dev is None:
                    sys_dev = find_system_audio_device(None)
                if sys_dev is not None:
                    q = queue.Queue(maxsize=64)
                    Recorder(sys_dev, q, self.stop_event, "시스템").start()
                    queues.append(["시스템", q])
                    self.ui_q.put(("status_add", f"시스템: {sys_dev.name}"))
                else:
                    self.ui_q.put(
                        (
                            "error",
                            "시스템 소리 장치를 찾지 못해 마이크로 대체합니다.",
                        )
                    )
                    if not any(lbl == "마이크" for lbl, _ in queues):
                        mic_dev = sc.default_microphone()
                        q = queue.Queue(maxsize=64)
                        Recorder(mic_dev, q, self.stop_event, "마이크").start()
                        queues.append(["마이크", q])
                        self.ui_q.put(("status_add", f"마이크: {mic_dev.name}"))

        out_path = args.out

        if is_mlx_streaming:
            self._run_mlx_streaming_loop(model, queues, out_path, args)
        else:
            self._run_chunk_loop(model, queues, out_path, args)

    # ---- 시스템 소리 권한이 없을 때: 백엔드를 끄지 않고 안내 한 번 + 조용히 재시도 ----
    SYSTEM_RETRY_SEC = 10.0

    def _on_system_channel_failed(self, err: str):
        self._sys_last_fail = time.time()
        if getattr(self, "_sys_fail_reported", False):
            return  # 같은 안내를 반복하지 않는다
        self._sys_fail_reported = True
        denied = "TCC" in err or "거절" in err or "declined" in err.lower()
        if denied:
            mic_note = (
                " 그동안 스피커 소리는 마이크로 잡혀 '마이크'로 표시될 수 있어요."
                if self.args.source == "both" else ""
            )
            self.ui_q.put((
                "error",
                "[시스템] 시스템 소리 녹음 권한이 꺼져 있어요. 방금 연 설정 화면 아래쪽 '시스템 오디오 녹음만'에서 "
                "Talkit 을 켜 주세요. 켜면 10초 안에 자동으로 다시 연결합니다." + mic_note,
            ))
            if platform.system() == "Darwin" and not os.environ.get("TALKIT_NO_OPEN_SETTINGS"):
                try:
                    import subprocess
                    subprocess.Popen(
                        ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    pass
        else:
            self.ui_q.put(("error", f"[시스템] 시스템 소리를 받을 수 없습니다: {err} (10초마다 다시 시도합니다)"))

    def _maybe_revive_system(self, queues):
        """시스템 채널이 빠져 있으면 10초마다 조용히 다시 연결해 본다 (권한을 켜면 저절로 붙음)."""
        if self.args.source not in ("system", "both"):
            return
        if any(lbl == "시스템" for lbl, _ in queues):
            return
        if time.time() - getattr(self, "_sys_last_fail", 0.0) < self.SYSTEM_RETRY_SEC:
            return
        tap_bin = find_macos_audio_tap_binary()
        if tap_bin is None:
            return
        self._sys_last_fail = time.time()
        q = queue.Queue(maxsize=64)
        ScreenCaptureKitRecorder(tap_bin, q, self.stop_event, "시스템").start()
        queues.append(["시스템", q])
        if getattr(self, "_sys_fail_reported", False):
            self.ui_q.put(("status", "시스템 소리 다시 연결 시도 중…"))

    def _read_channel_blocks(self, queues):
        self._maybe_revive_system(queues)
        if not queues:
            # 살아 있는 입력이 없어도 끝내지 않고 기다린다 (권한을 켜면 다시 붙는다)
            time.sleep(0.25)
            return {}
        blocks = {}
        dead_indices = []
        for idx, (label, q) in enumerate(queues):
            try:
                item = q.get(timeout=0.26 if not blocks else 0.04)
            except queue.Empty:
                continue
            if isinstance(item, Exception):
                if label == "시스템":
                    self._on_system_channel_failed(str(item))
                else:
                    self.ui_q.put(("error", str(item)))
                dead_indices.append(idx)
                continue
            blocks[label] = item

        for idx in reversed(dead_indices):
            queues.pop(idx)

        # 입력이 모두 빠져도 None(종료)을 돌려주지 않는다 — 다음 호출에서 다시 연결을 시도
        return blocks

    def _commit_text(
        self,
        text: str,
        out_path: str,
        utt_rms: float = 1.0,
        source_label: str | None = None,
    ):
        text = (text or "").strip()
        if not text:
            return
        # 문장 내 연속 구문 반복 루프("I'm a big set I'm a big set...", "너 이름은 뭐야? 너 이름은 뭐야?") 제거
        words = text.split()
        if len(words) >= 4:
            cleaned_words = []
            i = 0
            n = len(words)
            while i < n:
                collapsed = False
                for span in range(min(8, (n - i) // 2), 0, -1):
                    phrase = [w.strip(".,?!~") for w in words[i : i + span]]
                    nxt = [w.strip(".,?!~") for w in words[i + span : i + 2 * span]]
                    if phrase == nxt and (span >= 2 or ( i + 3 * span <= n and [w.strip(".,?!~") for w in words[i + 2 * span : i + 3 * span]] == phrase )):
                        cleaned_words.extend(words[i : i + span])
                        i += 2 * span
                        while i + span <= n and [w.strip(".,?!~") for w in words[i : i + span]] == phrase:
                            i += span
                        collapsed = True
                        break
                if not collapsed:
                    cleaned_words.append(words[i])
                    i += 1
            text = " ".join(cleaned_words).strip()

        last_map = getattr(self, "_last_committed_by_source", None)
        if last_map is None:
            last_map = {}
            self._last_committed_by_source = last_map
        key = source_label or "default"
        prev_committed = last_map.get(key, "")
        text = _stitch_overlap_text(prev_committed, text)
        if not text:
            return
        # 직전 확정 문장과 정규화 기준 완전히 동일하면서 저에너지(소음/숨소리)인 경우 중복 출력 차단
        norm_cur = "".join(ch for ch in text if ch.isalnum())
        norm_prev = "".join(ch for ch in prev_committed if ch.isalnum())
        if norm_cur and norm_cur == norm_prev and utt_rms < 0.020:
            return
        stripped = text.rstrip(".?!,~ ")
        compact_stripped = stripped.replace(" ", "")
        if (
            stripped in {"다", "요", "음", "응", "흠", "으", "그", "부", "이", "저"}
            or set(compact_stripped) <= {"음", "응", "어", "아", "으", "흠", "그"}
            or (utt_rms < 0.025 and len(compact_stripped) <= 1 and compact_stripped not in {"네", "예"})
        ):
            return
        last_map[key] = text
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        if source_label:
            line = f"[{stamp}] [{source_label}] {text}"
        else:
            line = f"[{stamp}] {text}"
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.ui_q.put(("text", line))

    def _run_mlx_streaming_loop(self, engine, queues, out_path, args):
        """
        고정밀 듀얼 채널 분리 실시간 스트리밍 파이프라인 (시스템 소리 / 마이크 독립 전사):
        1) 시스템(ScreenCaptureKit 디지털 원음)과 마이크(사용자 음성)를 합산(Mix)하지 않고
           각각 독립된 스트리밍 상태 머신 및 독립 VAD로 분리 전사
        2) 스피커 출력음이 마이크로 역유입(Bleed)될 때 마이크 채널에 동적 에코 게이트를 적용해
           시스템 소리가 마이크 카드로 중복 전사되거나 파형이 오염되는 현상을 원천 차단
        3) 공식 Qwen3-ASR 1.0초 청크 + unfixed_chunk_num=2 / unfixed_token_num=5 프리픽스 롤백
        """
        from collections import deque
        import gc
        import mlx.core as mx

        soft_split_sec = max(10.0, float(args.chunk) * 1.8)
        hard_ceiling_sec = max(16.0, float(args.chunk) * 2.6)

        def _make_channel():
            return {
                "state": None,
                "last_nonempty_partial": "",
                "recent_committed": deque(maxlen=10),
                "preroll_blocks": deque(maxlen=2),
                "band_history": deque(maxlen=80),
                "utterance_band_energies": [],
                "silent_blocks": 0,
                "active_sec": 0.0,
                "from_carryover": False,
                "new_speech_blocks": 0,
            }

        channels = {label: _make_channel() for label, _ in queues}
        sys_ref_ring = deque(maxlen=8)  # 최근 2.0초(8블록) 시스템 오디오 레퍼런스 링버퍼
        sys_active_holdover = 0  # 시스템 오디오 재생 중 + 종료 후 1.5초(6블록) 스피커 잔향 홀드오버
        mic_bleed_history = deque(maxlen=48)  # 스피커 재생 중 마이크로 유입되는 잔향 에너지 분포
        mic_direct_speech_streak = 0  # 사용자가 마이크 앞에서 직접 발화한 연속 블록 수
        step_counter = 0

        def _get_sys_reference_texts():
            sys_ch = channels.get("시스템")
            if not sys_ch:
                return []
            texts = list(sys_ch["recent_committed"])
            if sys_ch["last_nonempty_partial"]:
                texts.append(sys_ch["last_nonempty_partial"])
            if sys_ch["state"] is not None and getattr(sys_ch["state"], "text", ""):
                texts.append(sys_ch["state"].text)
            return texts

        def _init_new_state(carryover_audio=None, ch_obj=None):
            # Qwen3-ASR의 <system> context에 이전 전사 문장을 넣으면 저에너지/짧은 발화 구간에서
            # 시스템 프롬프트를 그대로 앵무새처럼 복창(Parroting)하는 현상이 발생하므로 context=""로 고정한다.
            st = engine.init_streaming_state(
                language="Korean",
                context="",
                chunk_size_sec=1.0,
                unfixed_chunk_num=4,
                unfixed_token_num=12,
            )
            if carryover_audio is not None and carryover_audio.size > 0:
                st.audio_accum = carryover_audio.astype(np.float32, copy=True)
            return st

        while not self.stop_event.is_set():
            blocks = self._read_channel_blocks(queues)
            if blocks is None:
                return
            if not blocks:
                continue

            if "시스템" in blocks:
                sys_ref_ring.append(blocks["시스템"].astype(np.float32, copy=True))

            max_level_norm = 0.0
            band_rms_map = {}
            raw_rms_map = {}
            for label, blk in blocks.items():
                r_rms = float(np.sqrt(np.mean(blk**2)))
                b_rms = _speech_band_rms(blk)
                raw_rms_map[label] = r_rms
                band_rms_map[label] = b_rms
                max_level_norm = max(max_level_norm, min(1.0, max(r_rms, b_rms) / 0.08))

            step_counter += 1
            if step_counter % 4 == 0:
                active_mb = mx.get_active_memory() / (1024 * 1024)
                cache_mb = mx.get_cache_memory() / (1024 * 1024)
                self.ui_q.put(
                    ("telemetry", (round(max_level_norm, 3), round(active_mb + cache_mb, 1)))
                )
            elif step_counter % 2 == 0:
                self.ui_q.put(("level", round(max_level_norm, 3)))

            sys_band_rms = band_rms_map.get("시스템", 0.0)
            sys_hist_band_rms = (
                max(_speech_band_rms(b) for b in sys_ref_ring)
                if len(sys_ref_ring) > 0
                else 0.0
            )
            effective_sys_rms = max(sys_band_rms, sys_hist_band_rms * 0.90)
            sys_ch_obj = channels.get("시스템")
            sys_state_open = (
                sys_ch_obj is not None and sys_ch_obj.get("state") is not None
            )
            if sys_band_rms > 0.0028 or effective_sys_rms > 0.0035 or sys_state_open:
                sys_active_holdover = 6  # 1.50초 동안 스피커 잔향 및 큐 지연 완전 차폐
            elif sys_active_holdover > 0:
                sys_active_holdover -= 1

            # 시스템 오디오를 먼저 처리하고 마이크를 처리한다
            ordered_labels = [lbl for lbl in ("시스템", "마이크") if lbl in blocks]
            for lbl in blocks:
                if lbl not in ordered_labels:
                    ordered_labels.append(lbl)

            for label in ordered_labels:
                block = blocks[label]
                if label not in channels:
                    channels[label] = _make_channel()
                ch = channels[label]

                band_rms = band_rms_map[label]
                ch["band_history"].append(band_rms)

                sub_len = max(1, block.shape[0] // 3)
                sub_max_band = max(
                    _speech_band_rms(block[:sub_len]),
                    _speech_band_rms(block[sub_len : 2 * sub_len]),
                    _speech_band_rms(block[2 * sub_len :]),
                )

                if len(ch["band_history"]) >= 8:
                    noise_floor = min(0.0085, float(np.percentile(ch["band_history"], 5)))
                else:
                    noise_floor = min(0.0075, args.silence * 0.5)

                speech_threshold = max(
                    0.0048, min(0.0105, noise_floor * 1.45, args.silence * 0.55)
                )
                # 시스템 오디오가 출력 중이거나 직전 1.5초 잔향 구간(sys_active_holdover > 0)일 때:
                # STFT 파형 왜곡 없이(뮤지컬 노이즈 환각 방지), 적응형 스피커 역유입 게이트(Adaptive Bleed Ducking Gate)로
                # 스피커 소리가 마이크 카드로 전사되는 것을 완벽 차단한다.
                is_echo_bleed = False
                if label == "마이크":
                    if sys_active_holdover > 0:
                        bleed_p95 = (
                            float(np.percentile(mic_bleed_history, 95))
                            if len(mic_bleed_history) >= 4
                            else 0.022
                        )
                        direct_voice_thresh = max(
                            0.058, bleed_p95 * 2.8, effective_sys_rms * 1.85
                        )
                        if (
                            band_rms >= direct_voice_thresh
                            and sub_max_band >= direct_voice_thresh * 1.15
                        ):
                            mic_direct_speech_streak += 1
                        else:
                            mic_direct_speech_streak = 0
                            if band_rms < 0.075:
                                mic_bleed_history.append(band_rms)

                        if mic_direct_speech_streak < 2:
                            is_echo_bleed = True
                            # 스피커 소리가 먼저 마이크를 살짝 열었을 경우 즉시 폐기하여 잔향 문장 확정 차단
                            if ch["state"] is not None and ch["active_sec"] < 2.25:
                                ch["state"] = None
                                ch["last_nonempty_partial"] = ""
                                ch["utterance_band_energies"].clear()
                                ch["preroll_blocks"].clear()
                                ch["silent_blocks"] = 0
                                ch["active_sec"] = 0.0
                                ch["from_carryover"] = False
                                ch["new_speech_blocks"] = 0
                                self.ui_q.put(("partial", f"[{label}] "))
                                continue
                    else:
                        mic_direct_speech_streak = 0

                is_silent = is_echo_bleed or (
                    (band_rms < speech_threshold)
                    and (sub_max_band < speech_threshold * 1.18)
                )

                state = ch["state"]
                active_sec = ch["active_sec"]
                utterance_band_energies = ch["utterance_band_energies"]

                if is_silent:
                    if state is None:
                        if not is_echo_bleed:
                            ch["preroll_blocks"].append(block)
                        continue

                    ch["silent_blocks"] += 1
                    required_pause_blocks = 2 if active_sec >= soft_split_sec else 3

                    if ch["silent_blocks"] < required_pause_blocks:
                        if not is_echo_bleed:
                            engine.streaming_transcribe(block, state)
                            if state.text:
                                if label == "마이크" and _is_text_echo_of_system(
                                    state.text, _get_sys_reference_texts()
                                ):
                                    self.ui_q.put(("partial", f"[{label}] "))
                                else:
                                    ch["last_nonempty_partial"] = state.text
                                    self.ui_q.put(("partial", f"[{label}] {state.text}"))
                        continue

                    utt_rms = (
                        float(np.median(utterance_band_energies))
                        if utterance_band_energies
                        else 1.0
                    )
                    # 1) hard_ceiling_sec 분할 직후 새로운 발화 없이 꼬리 무음으로 끝난 경우("습니다." 등) 중복 조각 폐기
                    # 2) 단일 블록(0.25초) 미세 소음/타건음(utt_rms < 0.012)으로 열린 경우 환각 방지 폐기
                    if (ch.get("from_carryover") and ch.get("new_speech_blocks", 0) <= 2) or (
                        not ch.get("from_carryover")
                        and ch.get("new_speech_blocks", 0) <= 1
                        and utt_rms < 0.012
                    ):
                        self.ui_q.put(("partial", f"[{label}] "))
                        ch["state"] = None
                        ch["last_nonempty_partial"] = ""
                        utterance_band_energies.clear()
                        ch["preroll_blocks"].clear()
                        ch["silent_blocks"] = 0
                        ch["active_sec"] = 0.0
                        ch["from_carryover"] = False
                        ch["new_speech_blocks"] = 0
                        mx.clear_cache()
                        gc.collect()
                        continue
                    try:
                        engine.finish_streaming_transcribe(state)
                        final_text = (
                            (state.text or "").strip()
                            or ch["last_nonempty_partial"].strip()
                        )
                        self.ui_q.put(("partial", f"[{label}] "))
                        if final_text and not (
                            label == "마이크"
                            and _is_text_echo_of_system(
                                final_text, _get_sys_reference_texts()
                            )
                        ):
                            self._commit_text(
                                final_text,
                                out_path,
                                utt_rms=utt_rms,
                                source_label=label,
                            )
                            ch["recent_committed"].append(final_text)
                    except Exception as e:
                        cand = ch["last_nonempty_partial"].strip()
                        if cand and not (
                            label == "마이크"
                            and _is_text_echo_of_system(cand, _get_sys_reference_texts())
                        ):
                            self._commit_text(
                                cand,
                                out_path,
                                utt_rms=utt_rms,
                                source_label=label,
                            )
                            ch["recent_committed"].append(cand)
                        self.ui_q.put(("error", f"[{label}] 스트리밍 마감 오류: {e}"))

                    ch["state"] = None
                    ch["last_nonempty_partial"] = ""
                    utterance_band_energies.clear()
                    ch["preroll_blocks"].clear()
                    if not is_echo_bleed:
                        ch["preroll_blocks"].append(block)
                    ch["silent_blocks"] = 0
                    ch["active_sec"] = 0.0
                    ch["from_carryover"] = False
                    ch["new_speech_blocks"] = 0
                    mx.clear_cache()
                    gc.collect()
                    continue

                ch["silent_blocks"] = 0
                ch["new_speech_blocks"] = ch.get("new_speech_blocks", 0) + 1
                utterance_band_energies.append(band_rms)

                if state is None:
                    ch["last_nonempty_partial"] = ""
                    ch["from_carryover"] = False
                    ch["new_speech_blocks"] = 1
                    carry = (
                        np.concatenate(list(ch["preroll_blocks"]))
                        if len(ch["preroll_blocks"]) > 0
                        else None
                    )
                    ch["preroll_blocks"].clear()
                    state = _init_new_state(carryover_audio=carry, ch_obj=ch)
                    ch["state"] = state

                if active_sec + BLOCK_SEC >= hard_ceiling_sec:
                    carryover = np.zeros((0,), dtype=np.float32)
                    try:
                        state.buffer = np.concatenate(
                            [state.buffer, block.astype(np.float32, copy=False)]
                        )
                        if state.buffer.shape[0] > 0:
                            state.audio_accum = np.concatenate(
                                [state.audio_accum, state.buffer]
                            )
                            state.buffer = np.zeros((0,), dtype=np.float32)

                        total_sec = state.audio_accum.shape[0] / float(SAMPLE_RATE)
                        fa_split = None
                        if hasattr(engine, "split_at_word_boundary") and getattr(engine, "forced_aligner", None) is not None:
                            fa_split = engine.split_at_word_boundary(
                                state,
                                min_sec=max(soft_split_sec - 2.0, total_sec - 5.0),
                                max_sec=max(soft_split_sec + 1.0, total_sec - 0.35),
                                sr=SAMPLE_RATE,
                            )

                        if fa_split is not None:
                            final_text, carryover = fa_split
                            final_text = (final_text or "").strip()
                        else:
                            split_idx = _find_valley_split_sample(
                                state.audio_accum,
                                min_sec=max(soft_split_sec, total_sec - 4.0),
                                max_sec=max(soft_split_sec + 1.0, total_sec - 0.35),
                                sr=SAMPLE_RATE,
                            )
                            if 0 < split_idx < state.audio_accum.shape[0]:
                                # 0.35초(5600샘플) 컨텍스트 오버랩을 carryover에 포함해 어절 절단("시키겠" / "습니다") 방지
                                overlap_start = max(0, split_idx - int(0.35 * SAMPLE_RATE))
                                carryover = state.audio_accum[overlap_start:].copy()
                                state.audio_accum = state.audio_accum[:split_idx]
                                setattr(state, "_valley_truncated", True)

                            engine.finish_streaming_transcribe(state)
                            final_text = (state.text or "").strip()

                        self.ui_q.put(("partial", f"[{label}] "))
                        if final_text and not (
                            label == "마이크"
                            and _is_text_echo_of_system(
                                final_text, _get_sys_reference_texts()
                            )
                        ):
                            self._commit_text(
                                final_text, out_path, source_label=label
                            )
                            ch["recent_committed"].append(final_text)
                    except Exception as e:
                        cand = ch["last_nonempty_partial"].strip()
                        if cand and not (
                            label == "마이크"
                            and _is_text_echo_of_system(cand, _get_sys_reference_texts())
                        ):
                            self._commit_text(
                                cand,
                                out_path,
                                source_label=label,
                            )
                            ch["recent_committed"].append(cand)
                        self.ui_q.put(
                            ("error", f"[{label}] 스트리밍 구간 확정 오류: {e}")
                        )

                    utterance_band_energies.clear()
                    ch["silent_blocks"] = 0
                    if carryover.size > 0:
                        ch["active_sec"] = carryover.shape[0] / float(SAMPLE_RATE)
                        ch["last_nonempty_partial"] = ""
                        ch["from_carryover"] = True
                        ch["new_speech_blocks"] = 0
                        ch["state"] = _init_new_state(carryover_audio=carryover, ch_obj=ch)
                    else:
                        ch["state"] = None
                        ch["last_nonempty_partial"] = ""
                        ch["active_sec"] = 0.0
                        ch["from_carryover"] = False
                        ch["new_speech_blocks"] = 0
                    mx.clear_cache()
                    gc.collect()
                    continue

                prev_chunk_id = state.chunk_id
                try:
                    engine.streaming_transcribe(block, state)
                    ch["active_sec"] += BLOCK_SEC
                    if state.text:
                        if label == "마이크" and _is_text_echo_of_system(
                            state.text, _get_sys_reference_texts()
                        ):
                            self.ui_q.put(("partial", f"[{label}] "))
                        else:
                            ch["last_nonempty_partial"] = state.text
                            if state.chunk_id != prev_chunk_id:
                                self.ui_q.put(
                                    ("partial", f"[{label}] {ch['last_nonempty_partial']}")
                                )
                except Exception as e:
                    self.ui_q.put(("error", f"[{label}] 스트리밍 전사 오류: {e}"))
                    ch["state"] = None
                    ch["last_nonempty_partial"] = ""
                    utterance_band_energies.clear()
                    ch["active_sec"] = 0.0
                    mx.clear_cache()
                    continue

    def _run_chunk_loop(self, model, queues, out_path, args):
        blocks_per_chunk = max(1, int(round(args.chunk / BLOCK_SEC)))
        while not self.stop_event.is_set():
            parts_by_label = {}
            for _ in range(blocks_per_chunk):
                blocks = self._read_channel_blocks(queues)
                if blocks is None:
                    return
                for label, blk in blocks.items():
                    parts_by_label.setdefault(label, []).append(blk)

            for label, parts in parts_by_label.items():
                chunk_audio = np.concatenate(parts)
                rms = float(np.sqrt(np.mean(chunk_audio**2)))
                if rms < args.silence:
                    continue
                peak = float(np.max(np.abs(chunk_audio)))
                if 0 < peak < 0.5:
                    chunk_audio = chunk_audio * (0.5 / peak)
                try:
                    text = transcribe_array(model, chunk_audio)
                except Exception as e:
                    self.ui_q.put(("error", f"[{label}] 전사 오류: {e}"))
                    continue
                self._commit_text(text, out_path, source_label=label)


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
