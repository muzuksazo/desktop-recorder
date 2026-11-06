"""
Talkit for Windows — exe 판의 바탕 기능: 경로, 엔진 설치, 모델 받기·양자화, 자동 업데이트.

Talkit.exe(PyInstaller, 화면만 들어 있음)는 무겁지 않다. 전사 엔진(Python + torch + qwen-asr)은 처음 실행할 때
%LOCALAPPDATA%\\Talkit 에 한 번 설치하고, 모델도 거기로 내려받는다. 소스에서 실행(python windows_app.py)하면
지금 쓰는 Python 과 저장소 폴더의 파일을 그대로 쓴다.

    %LOCALAPPDATA%\\Talkit\\
        runtime\\    Python 3.14.7 (python.org embeddable) + 패키지          ← 엔진 설치
        app\\        backend.py, live_transcribe.py, model_tool.py ...      ← exe 안의 것을 버전마다 풀어 둠
        models\\     내려받은 모델, 양자화해서 저장한 모델
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
import zipfile

APP_VERSION = "1.0.0"
GITHUB_REPO = "muzuksazo/desktop-recorder"
RELEASE_TAG_PREFIX = "win-v"          # Windows 판 릴리스 태그 (Mac 판은 v1.x.x)
EXE_ASSET = "Talkit.exe"

FROZEN = getattr(sys, "frozen", False)
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = getattr(sys, "_MEIPASS", HERE)
APP_HOME = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Talkit")
RUNTIME = os.path.join(APP_HOME, "runtime")
RUNTIME_PY = os.path.join(RUNTIME, "python.exe")
APP_DIR = os.path.join(APP_HOME, "app") if FROZEN else HERE
MODELS_DIR = os.path.join(APP_HOME, "models")
DEV_MODEL = os.path.normpath(os.path.join(HERE, "..", "qwen3_asr_1.7b_ko"))  # 소스 실행 때 옆에 받아 둔 모델

PY_VERSION = "3.14.7"
PY_ZIP = f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip"
GET_PIP = "https://bootstrap.pypa.io/get-pip.py"
TORCH = ["torch==2.11.0", "torchaudio==2.11.0"]
TORCH_INDEX = {"cuda": "https://download.pytorch.org/whl/cu128", "cpu": "https://download.pytorch.org/whl/cpu"}
BNB = "bitsandbytes==0.50.2"
APP_FILES = ["backend.py", "live_transcribe.py", "mlx_quantized_asr.py", "model_tool.py"]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# 받을 수 있는 모델 (전부 Apache-2.0)
MODELS = [
    {"id": "ko-1.7b", "repo": "TeamUNIVA/qwen3_asr_1.7b_ko", "folder": "qwen3_asr_1.7b_ko", "gb": 4.7,
     "label": "한국어 1.7B (기본, 한국어 특화)"},
    {"id": "qwen-1.7b", "repo": "Qwen/Qwen3-ASR-1.7B", "folder": "Qwen3-ASR-1.7B", "gb": 4.7,
     "label": "Qwen 공식 1.7B (여러 언어)"},
    {"id": "qwen-0.6b", "repo": "Qwen/Qwen3-ASR-0.6B", "folder": "Qwen3-ASR-0.6B", "gb": 1.9,
     "label": "Qwen 공식 0.6B (가벼움, GPU 없는 PC용)"},
]

# 불러올 때의 정밀도 (양자화)
QUANTS = [
    ("auto", "자동 (GPU 메모리를 보고 고름)"),
    ("none", "원본 (GPU bf16 / CPU fp32)"),
    ("int8", "8-bit (GPU, bitsandbytes — 메모리 절반, 하지만 느림)"),
    ("nf4", "4-bit (GPU, bitsandbytes — 메모리 1/3, 속도 같음)"),
    ("cpu-int8", "CPU 8-bit (GPU 없는 PC)"),
]


def _ver(s):
    return tuple(int(x) for x in re.findall(r"\d+", s)[:3])


# ---------------------------------------------------------------- 하드웨어
_gpu = None


def nvidia_gpu():
    """(이름, 메모리 MB) 또는 None. nvidia-smi 로 본다 (torch 를 불러오지 않음)."""
    global _gpu
    if _gpu is None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW).stdout
            name, mb = out.strip().splitlines()[0].rsplit(",", 1)
            _gpu = (name.strip(), int(mb))
        except Exception:
            _gpu = False
    return _gpu or None


def auto_quant():
    """GPU 6GB 이상 → 원본(bf16), GPU 그보다 작으면 4-bit, GPU 없으면 CPU 8-bit"""
    g = nvidia_gpu()
    if g is None:
        return "cpu-int8"
    return "none" if g[1] >= 6000 else "nf4"


# ---------------------------------------------------------------- 엔진 (Python + 패키지)
def engine_python():
    if os.environ.get("RECORDER_PYTHON"):
        return os.environ["RECORDER_PYTHON"]
    return RUNTIME_PY if FROZEN else sys.executable


def _requirements_path():
    p = os.path.join(BUNDLE, "requirements-windows.txt")
    return p if os.path.exists(p) else os.path.join(HERE, "packaging", "requirements-windows.txt")


def _engine_key():
    """설치할 것이 바뀌었는지 알기 위한 지문 (패키지 목록 + GPU 여부)"""
    with open(_requirements_path(), "rb") as f:
        h = hashlib.sha256(f.read())
    h.update(("cuda" if nvidia_gpu() else "cpu").encode())
    h.update(" ".join(TORCH + [BNB, PY_VERSION]).encode())
    return h.hexdigest()[:16]


def engine_ready():
    if not FROZEN:
        return True  # 소스 실행: 지금 Python 을 그대로 쓴다
    try:
        with open(os.path.join(RUNTIME, "talkit_engine.txt"), encoding="utf-8") as f:
            return f.read().strip() == _engine_key() and os.path.exists(RUNTIME_PY)
    except OSError:
        return False


def _download(url, path, log, label):
    req = urllib.request.Request(url, headers={"User-Agent": "Talkit-Windows"})
    with urllib.request.urlopen(req, timeout=60) as r, open(path + ".part", "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, -1
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            pct = done * 100 // total if total else -1
            if pct != last and pct % 5 == 0:
                last = pct
                log(f"{label} 받는 중… {pct}%" if total else f"{label} 받는 중… {done >> 20}MB")
    os.replace(path + ".part", path)


def _run(cmd, log, cwd=None):
    """명령을 실행하고 출력을 한 줄씩 log 로 넘긴다. 실패하면 예외."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")  # 한글 출력이 깨지지 않게
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", creationflags=NO_WINDOW, env=env)
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        if line:
            tail = (tail + [line])[-15:]
            log(line)
    if p.wait() != 0:
        raise RuntimeError(f"실패 (코드 {p.returncode}): {' '.join(cmd[:4])}…\n" + "\n".join(tail[-5:]))


def install_engine(log):
    """전사 엔진을 %LOCALAPPDATA%\\Talkit\\runtime 에 설치한다 (처음 한 번, 몇 GB, 10~30분)."""
    os.makedirs(RUNTIME, exist_ok=True)
    gpu = nvidia_gpu()
    log(f"GPU: {gpu[0]} ({round(gpu[1] / 1024)}GB)" if gpu else "NVIDIA GPU 없음 → CPU 판으로 설치")

    if not os.path.exists(RUNTIME_PY):
        zp = os.path.join(APP_HOME, "python-embed.zip")
        _download(PY_ZIP, zp, log, f"Python {PY_VERSION}")
        with zipfile.ZipFile(zp) as z:
            z.extractall(RUNTIME)
        os.remove(zp)
    # embeddable Python 은 site-packages 와 스크립트 폴더를 기본으로 안 본다 → ._pth 에 적는다
    pth = next(f for f in os.listdir(RUNTIME) if f.endswith("._pth"))
    zipname = next(f for f in os.listdir(RUNTIME) if re.fullmatch(r"python\d+\.zip", f))
    with open(os.path.join(RUNTIME, pth), "w", encoding="utf-8") as f:
        f.write(f"{zipname}\n.\nLib\\site-packages\n{APP_DIR}\nimport site\n")

    if subprocess.run([RUNTIME_PY, "-m", "pip", "--version"], capture_output=True,
                      creationflags=NO_WINDOW).returncode != 0:
        gp = os.path.join(RUNTIME, "get-pip.py")
        _download(GET_PIP, gp, log, "pip")
        _run([RUNTIME_PY, gp, "--no-warn-script-location"], log)

    pip = [RUNTIME_PY, "-m", "pip", "install", "--no-warn-script-location", "--prefer-binary",
           "--disable-pip-version-check"]
    kind = "cuda" if gpu else "cpu"
    log(f"torch ({kind}) 설치 중… (가장 큼, {'약 3GB' if gpu else '약 200MB'})")
    _run(pip + TORCH + ["--index-url", TORCH_INDEX[kind]], log)
    log("나머지 패키지 설치 중…")
    _run(pip + ["-r", _requirements_path()] + ([BNB] if gpu else []), log)
    sync_app_files()
    _run([RUNTIME_PY, "-c", "import torch, qwen_asr, silero_vad, soundcard; print('엔진 확인 OK · torch', "
                            "torch.__version__, '· CUDA', torch.cuda.is_available())"], log, cwd=APP_DIR)
    with open(os.path.join(RUNTIME, "talkit_engine.txt"), "w", encoding="utf-8") as f:
        f.write(_engine_key())
    log("✅ 엔진 설치 끝")


def sync_app_files():
    """exe 안에 든 엔진 스크립트를 %LOCALAPPDATA%\\Talkit\\app 에 풀어 둔다 (버전이 바뀌면 다시)."""
    if not FROZEN:
        return
    os.makedirs(APP_DIR, exist_ok=True)
    for n in APP_FILES:  # 내용이 다를 때만 바꾼다 (같은 버전 번호로 고친 파일도 반영되게)
        src, dst = os.path.join(BUNDLE, "payload", n), os.path.join(APP_DIR, n)
        with open(src, "rb") as f:
            new = f.read()
        try:
            with open(dst, "rb") as f:
                if f.read() == new:
                    continue
        except OSError:
            pass
        with open(dst, "wb") as f:
            f.write(new)
    with open(os.path.join(APP_DIR, "version.txt"), "w", encoding="utf-8") as f:
        f.write(APP_VERSION)


# ---------------------------------------------------------------- 모델
def is_model_dir(p):
    return os.path.isfile(os.path.join(p, "config.json")) and any(
        n.endswith(".safetensors") for n in os.listdir(p))


def installed_models():
    """[(표시 이름, 폴더)] — 받아 둔 모델과 양자화해서 저장한 모델"""
    out = []
    if not FROZEN and os.path.isdir(DEV_MODEL) and is_model_dir(DEV_MODEL):
        out.append((f"{os.path.basename(DEV_MODEL)} (저장소 옆 폴더)", DEV_MODEL))
    if os.path.isdir(MODELS_DIR):
        for n in sorted(os.listdir(MODELS_DIR)):
            p = os.path.join(MODELS_DIR, n)
            if os.path.isdir(p) and is_model_dir(p):
                q = ""
                try:
                    with open(os.path.join(p, "config.json"), encoding="utf-8") as f:
                        qc = json.load(f).get("quantization_config")
                    if qc:
                        q = " · 4-bit" if qc.get("load_in_4bit") else " · 8-bit"
                except Exception:
                    pass
                out.append((n + q, p))
    return out


def dir_size(p):
    total = 0
    for root, _, files in os.walk(p):
        for n in files:
            try:
                total += os.path.getsize(os.path.join(root, n))
            except OSError:
                pass
    return total


def run_model_tool(args, on_msg):
    """엔진 Python 으로 model_tool.py 를 실행한다. JSON 한 줄씩 on_msg 로. 끝나면 종료 코드를 돌려준다."""
    sync_app_files()
    p = subprocess.Popen([engine_python(), "-X", "utf8", os.path.join(APP_DIR, "model_tool.py"), *args],
                         cwd=APP_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    for line in p.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            on_msg(json.loads(line))
        except json.JSONDecodeError:
            on_msg({"type": "log", "msg": line})
    return p.wait()


# ---------------------------------------------------------------- 자동 업데이트 (GitHub 릴리스)
def check_update():
    """더 새 Windows 판이 있으면 {'version', 'url', 'notes', 'page'} 를, 없으면 None 을 돌려준다."""
    req = urllib.request.Request(f"https://api.github.com/repos/{GITHUB_REPO}/releases?per_page=30",
                                 headers={"User-Agent": "Talkit-Windows", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        releases = json.load(r)
    best = None
    for rel in releases:
        tag = rel.get("tag_name", "")
        if rel.get("draft") or rel.get("prerelease") or not tag.startswith(RELEASE_TAG_PREFIX):
            continue
        asset = next((a for a in rel.get("assets", []) if a.get("name") == EXE_ASSET), None)
        if asset and (best is None or _ver(tag) > _ver(best["version"])):
            best = {"version": tag[len(RELEASE_TAG_PREFIX):], "url": asset["browser_download_url"],
                    "notes": rel.get("body") or "", "page": rel.get("html_url")}
    if best and _ver(best["version"]) > _ver(APP_VERSION):
        return best
    return None


def apply_update(info, log):
    """새 Talkit.exe 를 받아 지금 exe 옆에 두고, 앱이 꺼지면 바꿔 끼운 뒤 다시 켜는 스크립트를 띄운다."""
    if not FROZEN:
        raise RuntimeError("소스에서 실행 중이에요. git pull 로 업데이트하세요.")
    exe = sys.executable
    new = exe + ".new"
    _download(info["url"], new, log, f"Talkit {info['version']}")
    with open(new, "rb") as f:
        if f.read(2) != b"MZ":
            raise RuntimeError("받은 파일이 실행 파일이 아니에요.")
    cmd = os.path.join(APP_HOME, "update.cmd")
    with open(cmd, "w", encoding="utf-8") as f:
        f.write(
            "@echo off\r\nchcp 65001 >nul\r\n"
            f":wait\r\ntasklist /FI \"PID eq {os.getpid()}\" | find \"{os.getpid()}\" >nul && "
            "(timeout /t 1 /nobreak >nul & goto wait)\r\n"
            f"move /y \"{new}\" \"{exe}\" >nul\r\n"
            f"start \"\" \"{exe}\"\r\n"
        )
    subprocess.Popen(["cmd", "/c", cmd], creationflags=NO_WINDOW | getattr(subprocess, "DETACHED_PROCESS", 0),
                     close_fds=True)
    log("업데이트 준비 끝 — 앱을 다시 시작합니다")
