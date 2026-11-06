"""
Talkit for Windows (tkinter) — Mac 앱 Talkit 1.2.0 의 Windows 판 + Windows 기본 음성 읽기(TTS).

Rust 앱(app/)을 빌드할 수 없는 PC(스마트 앱 컨트롤이 켜진 Windows)용이다. 추가 설치가 필요 없다.

Talkit 1.2.0 과 같은 것
- 녹음 소스 선택: 마이크 / 시스템 / 둘 다 (바꾸면 백엔드를 다시 띄우고 같은 파일에 이어 쓴다)
- 전사 따라가기 켜고 끄기 (끄면 위로 올려 읽는 동안 강제로 내려가지 않음)
- 이전 녹음 기록 목록과 다시 보기 (문서\\Talkit\\transcript_*.txt)
- 시스템 / 마이크 출처 표시, 문장 단위로 나눠 보이기
- 일시정지·재개, 녹음 시간, 소리 크기 막대, 드래그해서 복사, 전체복사, 비우기, 미니창
- 설정 기억 (%APPDATA%\\Talkit\\settings.txt)

Windows 판에만 있는 것
- TTS: PowerShell System.Speech(SAPI)로 한국어 음성(Microsoft Heami)이 읽는다. 문장을 누르면 읽고,
  "자동 읽기"를 켜면 새 문장마다 읽는다. TTS 소리가 다시 전사되면 걸러 낸다 (app/src/tts.rs 와 같은 규칙).

실행
    python windows_app.py
    python windows_app.py --source system --topmost --auto-read
    python windows_app.py --input-wav a.wav     # 녹음 대신 파일로 시험
그 밖의 옵션(--chunk, --context, --pause 등)은 그대로 backend.py 로 넘어간다.
"""

import argparse
import base64
import datetime as dt
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from collections import deque
from tkinter import messagebox, simpledialog, ttk

import talkit_win as tw

HERE = tw.APP_DIR          # backend.py 가 있는 곳 (exe 판은 %LOCALAPPDATA%\Talkit\app)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
ECHO_MARGIN = 1.0          # 메아리 판단 때 앞뒤 여유(초)
MAX_ITEMS = 300            # 화면에 남겨 둘 문장 수
SOURCES = [("mic", "마이크"), ("system", "시스템"), ("both", "둘 다")]

# Talkit 색
BLUE, BLACK, GRAY, WHITE = "#4f6bff", "#1a1a1a", "#e9ebf7", "#ffffff"
MUTED, RED, SYS_BG, MIC_BG = "#7a7f99", "#c62828", "#e3e8ff", "#e6f4ea"
FONT = "Malgun Gothic"


# ---------------------------------------------------------------- 경로·설정
def documents_dir():
    """사용자의 '문서' 폴더 (OneDrive 로 옮겨 둔 경우도 따라감)"""
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(260)
        if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0:  # 5 = CSIDL_PERSONAL
            return buf.value
    except Exception:
        pass
    return os.path.join(os.path.expanduser("~"), "Documents")


TALKIT_DOCS = os.path.join(documents_dir(), "Talkit")
SETTINGS = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "Talkit", "settings.txt")


def read_settings():
    try:
        with open(SETTINGS, encoding="utf-8") as f:
            return dict(l.strip().split("=", 1) for l in f if "=" in l)
    except OSError:
        return {}


def write_setting(key, value):
    s = read_settings()
    s[key] = str(value)
    try:
        os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
        with open(SETTINGS, "w", encoding="utf-8") as f:
            f.writelines(f"{k}={v}\n" for k, v in s.items())
    except OSError:
        pass


# ---------------------------------------------------------------- 글 다루기 (app/src/main.rs 와 같은 규칙)
ENDINGS = ("습니다", "ㅂ니다", "입니다", "였습니다", "겠습니다", "랍니다",
           "네요", "군요", "어요", "아요", "해요", "예요", "이에요", "거예요",
           "까요", "나요", "가요", "지요", "죠", "세", "십시오",
           "했다", "됐다", "한다", "된다", "있다", "없다", "이다", "였다", "겠다")


def is_sentence_end(word, count):
    w = word.rstrip("\"')]”’")
    if not w:
        return False
    if w[-1] in "?!…":
        return True
    if w[-1] == ".":
        stem = w.rstrip(".")
        if not (stem and stem.isdigit()):  # "3." 같은 번호는 문장 끝이 아님
            return True
    if count >= 3:
        clean = w.rstrip("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        return clean.endswith(ENDINGS)
    return False


def split_sentences(text):
    """[(문장, 끝났는가)] — 한 번에 여러 문장이 오면 나눈다"""
    out, cur = [], []
    for w in text.split():
        cur.append(w)
        if is_sentence_end(w, len(cur)):
            out.append((" ".join(cur), True))
            cur = []
    if cur:
        out.append((" ".join(cur), False))
    return out


def split_source(s):
    """'[시스템] 문장' → ('시스템', '문장')"""
    s = s.lstrip()
    for src in ("시스템", "마이크"):
        if s.startswith(f"[{src}]"):
            return src, s[len(src) + 2:].lstrip()
    return None, s


def parse_history_line(line):
    """'[14:31:05] [시스템] 문장' → (시각, 출처, 문장)"""
    line = line.strip()
    clock, rest = None, line
    if line.startswith("[") and "] " in line:
        tag, after = line[1:].split("] ", 1)
        if tag not in ("시스템", "마이크"):
            clock, rest = tag, after
    src, body = split_source(rest)
    return (clock, src, body.strip()) if body.strip() else None


def history_title(path):
    """transcript_2026-11-05_1740.txt → 2026-11-05 17:40"""
    core = os.path.splitext(os.path.basename(path))[0].removeprefix("transcript_")
    if "_" in core:
        d, hm = core.split("_", 1)
        if len(hm) == 4 and hm.isdigit():
            return f"{d} {hm[:2]}:{hm[2:]}"
    return core


def scan_history(extra_dir=None):
    seen, out = set(), []
    for d in filter(None, [TALKIT_DOCS, extra_dir, HERE]):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            if not (n.startswith("transcript_") and n.endswith(".txt")):
                continue
            p = os.path.join(d, n)
            key = os.path.normcase(os.path.realpath(p))
            if key in seen:
                continue
            seen.add(key)
            lines = read_history(p)
            preview = lines[0][2][:60] if lines else "(내용 없음)"
            out.append({"path": p, "title": history_title(p), "count": len(lines), "preview": preview})
    out.sort(key=lambda e: os.path.basename(e["path"]), reverse=True)  # 이름이 시간순 → 역순 = 최신순
    return out


def read_history(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return [x for x in map(parse_history_line, f) if x]
    except OSError:
        return []


# ---------------------------------------------------------------- 메아리 판단
def similarity(a, b):
    """두 문장이 얼마나 비슷한가 (0~1, 공백·기호 무시 글자 2개 묶음 기준)"""

    def bigrams(s):
        cs = [c for c in s if c.isalnum()]
        return [(cs[i], cs[i + 1]) for i in range(len(cs) - 1)]

    x, y = bigrams(a), bigrams(b)
    if not x or not y:
        return 0.0
    total, common = len(x) + len(y), 0
    for g in x:
        if g in y:
            y.remove(g)
            common += 1
    return 2.0 * common / total


# ---------------------------------------------------------------- TTS
PS_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$v = $s.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -eq 'ko-KR' } | Select-Object -First 1
if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }
[Console]::Out.WriteLine('READY ' + $s.Voice.Name + $(if ($v) { '' } else { ' (한국어 음성 없음)' }))
[Console]::Out.Flush()
while ($null -ne ($l = [Console]::In.ReadLine())) {
    $s.Speak([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($l)))
    [Console]::Out.WriteLine('DONE')
    [Console]::Out.Flush()
}
"""


class Tts:
    """한 번에 한 문장씩 차례로 읽는다. 읽은 구간(시작, 끝, 글)을 기록해 메아리를 거른다."""

    def __init__(self, on_ready):
        self.on_ready = on_ready
        self.lock = threading.Lock()
        self.queue = []
        self.spans = []  # [시작, 끝(읽는 중이면 inf), 글]
        self.proc = None
        self.busy = False
        self.ready = threading.Event()
        self._spawn()

    def _spawn(self):
        self.ready.clear()
        enc = base64.b64encode(PS_SCRIPT.encode("utf-16-le")).decode()
        try:
            proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-EncodedCommand", enc],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                creationflags=NO_WINDOW, text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
        except OSError as e:
            self.on_ready(False, f"TTS 사용 불가: {e}")
            return
        self.proc = proc
        threading.Thread(target=self._read, args=(proc,), daemon=True).start()

    def _read(self, proc):
        for line in proc.stdout:
            line = line.strip()
            if proc is not self.proc:  # 멈춤으로 버린 프로세스가 늦게 보낸 줄
                return
            if line.startswith("READY"):
                self.on_ready(True, line[6:])
                self.ready.set()
                self._next()
            elif line == "DONE":
                with self.lock:
                    self._close_span()
                    self.busy = False
                self._next()

    def _close_span(self):
        if self.spans and self.spans[-1][1] == float("inf"):
            self.spans[-1][1] = time.time()

    def _next(self):
        with self.lock:
            if self.busy or not self.queue or not self.ready.is_set() or self.proc is None:
                return
            text = self.queue.pop(0)
            self.busy = True
            self.spans.append([time.time(), float("inf"), text])
            try:
                self.proc.stdin.write(base64.b64encode(text.encode("utf-8")).decode() + "\n")
                self.proc.stdin.flush()
            except OSError:
                self.busy = False
                self._close_span()

    def speak(self, text):
        with self.lock:
            self.queue.append(text)
        self._next()

    def stop(self):
        """읽던 것과 대기열을 모두 버린다 (PowerShell 을 끄고 새로 띄움)."""
        with self.lock:
            self.queue.clear()
            self._close_span()
            self.busy = False
            old, self.proc = self.proc, None
        if old is not None:
            old.kill()
        self._spawn()

    def close(self):
        with self.lock:
            self.queue.clear()
            old, self.proc = self.proc, None
        if old is not None:
            old.kill()

    def is_echo(self, text, start, end, margin=ECHO_MARGIN):
        """[start, end] 동안 읽던 글과 전사된 글이 비슷하면 TTS 소리가 다시 녹음된 것으로 본다."""
        with self.lock:
            cutoff = time.time() - 120
            self.spans = [s for s in self.spans if s[1] > cutoff]
            spoken = " ".join(t for s, e, t in self.spans if start < e + margin and end > s - margin)
        return bool(spoken) and (similarity(text, spoken) >= 0.5 or text.strip() in spoken)


# ---------------------------------------------------------------- 백엔드
def start_backend(python, backend_args, ui_q, gen):
    cmd = [python, "-X", "utf8", os.path.join(HERE, "backend.py"), *backend_args]
    try:
        proc = subprocess.Popen(
            cmd, cwd=HERE, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=None,
            creationflags=NO_WINDOW if python.lower().endswith("pythonw.exe") else 0,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
    except OSError as e:
        ui_q.put((gen, {"type": "error", "msg": f"백엔드를 실행할 수 없습니다: {e}"}))
        return None

    def read():
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ui_q.put((gen, json.loads(line)))
            except json.JSONDecodeError:
                ui_q.put((gen, {"type": "status", "msg": line}))
        ui_q.put((gen, {"type": "exited", "code": proc.wait()}))

    threading.Thread(target=read, daemon=True).start()
    return proc


# ---------------------------------------------------------------- 작은 위젯
def pill(parent, text, cmd, primary=False):
    b = tk.Button(parent, text=text, command=cmd, relief="flat", bd=0, padx=10, pady=3,
                  font=(FONT, 9, "bold" if primary else "normal"), cursor="hand2",
                  bg=BLUE if primary else GRAY, fg=WHITE if primary else BLACK,
                  activebackground="#3d56e0" if primary else "#d8dbee",
                  activeforeground=WHITE if primary else BLACK)
    return b


def check(parent, text, var, cmd=None):
    return tk.Checkbutton(parent, text=text, variable=var, command=cmd, bg=WHITE, fg=BLACK,
                          activebackground=WHITE, selectcolor=WHITE, font=(FONT, 9), cursor="hand2")


class Wave(tk.Canvas):
    """최근 소리 크기 막대 12개"""

    def __init__(self, parent, bars=12, h=22):
        super().__init__(parent, width=bars * 6, height=h, bg=WHITE, highlightthickness=0)
        self.h, self.hist = h, deque([0.06] * bars, maxlen=bars)

    def push(self, level):
        self.hist.append(max(0.06, min(1.0, level)))

    def draw(self, active):
        self.delete("all")
        for i, v in enumerate(self.hist):
            bh = max(2, v * self.h)
            y0 = (self.h - bh) / 2
            self.create_rectangle(i * 6, y0, i * 6 + 3, y0 + bh, width=0, fill=BLUE if active else "#c5cae9")


# ---------------------------------------------------------------- 창
class App:
    def __init__(self, root, args, backend_args):
        self.root, self.args, self.backend_args = root, args, backend_args
        st = read_settings()
        self.ui_q = queue.Queue()
        self.gen = 0
        self.backend = None
        self.items = []          # {ts, src, body, done, err}
        self.line_items = []     # 화면 줄 번호(0부터) → items 번호 (없으면 None)
        self.echo_dropped = 0
        self.last_infer = None
        self.voice = "음성 준비 중…"
        self.status = "백엔드 시작 중…"
        self.toast, self.toast_until = "", 0.0
        self.paused = False
        self.level = 0.0
        self.started = time.time()
        self.mouse_down = False
        self.dirty = False
        self.view = "live"       # live / list / detail
        self.history, self.detail = [], None
        self.mini = False
        self.studio_geom = None

        self.source = tk.StringVar(value=args.source or st.get("source", "both"))
        self.follow = tk.BooleanVar(value=st.get("follow", "1") == "1")
        self.auto_read = tk.BooleanVar(value=args.auto_read or st.get("auto_read") == "1")
        self.topmost = tk.BooleanVar(value=args.topmost or st.get("topmost") == "1")
        self.out_path = args.out or os.path.join(
            TALKIT_DOCS, dt.datetime.now().strftime("transcript_%Y-%m-%d_%H%M.txt"))
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)

        root.title("Talkit")
        root.geometry("860x600")
        root.minsize(420, 150)
        root.configure(bg=WHITE)
        root.attributes("-topmost", self.topmost.get())
        self.build_studio()
        self.build_mini()
        self.studio.pack(fill="both", expand=True)

        self.tts = Tts(lambda ok, msg: self.ui_q.put((None, {"type": "tts_ready", "ok": ok, "msg": msg})))
        self.update_info = None
        self.settings_win = None
        self.start_backend()
        if self.backend is None:
            root.after(300, self.open_settings)  # 엔진·모델이 없으면 설정부터
        threading.Thread(target=self.bg_check_update, daemon=True).start()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Control-c>", lambda e: self.copy_selection())
        self.poll()

    # ================================================================ 화면 만들기
    def build_studio(self):
        r = self.root
        self.studio = tk.Frame(r, bg=WHITE)

        head = tk.Frame(self.studio, bg=WHITE)
        head.pack(fill="x", padx=14, pady=(12, 4))
        tk.Label(head, text="Talkit", font=(FONT, 16, "bold"), fg=BLUE, bg=WHITE).pack(side="left")
        tk.Label(head, text="  말하는 순간, 바로 텍스트로.", font=(FONT, 9), fg=MUTED, bg=WHITE).pack(side="left")
        pill(head, "미니창", self.toggle_mini).pack(side="right")
        pill(head, "설정", self.open_settings).pack(side="right", padx=4)
        self.update_btn = pill(head, "", self.open_settings, primary=True)  # 새 버전이 있을 때만 보임
        self.rec = tk.Label(head, font=(FONT, 10, "bold"), bg=WHITE)
        self.rec.pack(side="right", padx=10)
        self.wave = Wave(head)
        self.wave.pack(side="right")

        bar = tk.Frame(self.studio, bg=WHITE)
        bar.pack(fill="x", padx=14, pady=2)
        tk.Label(bar, text="입력", font=(FONT, 9, "bold"), fg=MUTED, bg=WHITE).pack(side="left")
        for val, label in SOURCES:
            tk.Radiobutton(bar, text=label, value=val, variable=self.source, command=self.on_source,
                           indicatoron=False, relief="flat", bd=0, padx=10, pady=3, font=(FONT, 9),
                           bg=GRAY, selectcolor=BLUE, activebackground="#d8dbee", cursor="hand2",
                           fg=BLACK).pack(side="left", padx=(6 if val == "mic" else 2, 0))
        self.pause_btn = pill(bar, "일시정지", self.toggle_pause)
        self.pause_btn.pack(side="left", padx=(14, 4))
        check(bar, "따라가기", self.follow, self.on_follow).pack(side="left")
        self.hist_btn = pill(bar, "기록", self.open_history)
        self.hist_btn.pack(side="right")

        bar2 = tk.Frame(self.studio, bg=WHITE)
        bar2.pack(fill="x", padx=14, pady=(2, 6))
        check(bar2, "자동 읽기", self.auto_read, lambda: write_setting("auto_read", int(self.auto_read.get()))
              ).pack(side="left")
        pill(bar2, "읽기 멈춤", lambda: self.tts.stop()).pack(side="left", padx=4)
        check(bar2, "항상 위", self.topmost, self.on_topmost).pack(side="left", padx=(6, 0))
        pill(bar2, "비우기", self.clear).pack(side="right")
        pill(bar2, "전체복사", self.copy_all).pack(side="right", padx=4)
        tk.Label(bar2, text="문장을 누르면 읽고, 끌어서 고르면 복사돼요", font=(FONT, 8), fg=MUTED, bg=WHITE
                 ).pack(side="right", padx=8)

        # 기록 화면용 줄 (기록 볼 때만 보임)
        self.hbar = tk.Frame(self.studio, bg=WHITE)
        pill(self.hbar, "← 실시간", self.close_history, primary=True).pack(side="left")
        self.hbar_list = pill(self.hbar, "← 목록", self.open_history)
        self.hbar_refresh = pill(self.hbar, "새로고침", self.open_history)
        self.hbar_copy = pill(self.hbar, "전체복사", self.copy_all)
        self.htitle = tk.Label(self.hbar, font=(FONT, 10, "bold"), fg=BLACK, bg=WHITE)
        self.htitle.pack(side="left", padx=10)

        body = tk.Frame(self.studio, bg=GRAY)
        body.pack(fill="both", expand=True, padx=14)
        self.text = tk.Text(body, wrap="word", font=(FONT, 12), bd=0, padx=12, pady=8, bg=WHITE, fg=BLACK,
                            cursor="hand2", state="disabled", spacing1=4, spacing3=4,
                            selectbackground="#c9d2ff", inactiveselectbackground="#c9d2ff")
        sb = tk.Scrollbar(body, command=self.text.yview)
        self.text.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True, padx=1, pady=1)
        t = self.text
        t.tag_configure("ts", foreground=MUTED, font=(FONT, 9))
        t.tag_configure("시스템", background=SYS_BG, foreground=BLUE, font=(FONT, 9, "bold"))
        t.tag_configure("마이크", background=MIC_BG, foreground="#2e7d32", font=(FONT, 9, "bold"))
        t.tag_configure("err", foreground=RED)
        t.tag_configure("open", foreground="#444a66")
        t.tag_configure("partial", foreground=MUTED, font=(FONT, 12, "italic"))
        t.tag_configure("htitle", font=(FONT, 11, "bold"))
        t.tag_configure("hover", background="#f1f3fc")
        t.tag_raise("sel")
        t.bind("<ButtonPress-1>", lambda e: setattr(self, "mouse_down", True))
        t.bind("<ButtonRelease-1>", self.on_release)
        t.bind("<Motion>", self.on_motion)
        t.bind("<Leave>", lambda e: t.tag_remove("hover", "1.0", "end"))

        self.info = tk.Label(self.studio, anchor="w", fg=MUTED, bg=WHITE, justify="left", font=(FONT, 8))
        self.info.pack(fill="x", padx=14, pady=(4, 10))

    def build_mini(self):
        self.minif = tk.Frame(self.root, bg=WHITE)
        top = tk.Frame(self.minif, bg=WHITE)
        top.pack(fill="x", padx=12, pady=(8, 2))
        tk.Label(top, text="Talkit", font=(FONT, 10, "bold"), fg=BLUE, bg=WHITE).pack(side="left")
        self.mini_rec = tk.Label(top, font=(FONT, 9, "bold"), bg=WHITE)
        self.mini_rec.pack(side="left", padx=8)
        self.mini_wave = Wave(top, h=16)
        self.mini_wave.pack(side="left")
        pill(top, "큰창", self.toggle_mini, primary=True).pack(side="right")
        self.mini_pause = pill(top, "일시정지", self.toggle_pause)
        self.mini_pause.pack(side="right", padx=4)
        self.mini_text = tk.Label(self.minif, anchor="sw", justify="left", font=(FONT, 13), fg=BLACK, bg=WHITE,
                                  wraplength=480)
        self.mini_text.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.minif.bind("<Configure>", lambda e: self.mini_text.config(wraplength=max(200, e.width - 24)))

    # ================================================================ 백엔드
    def start_backend(self):
        self.gen += 1
        if not tw.engine_ready():
            self.status = "전사 엔진이 아직 설치되지 않았어요 — 설정에서 '엔진 설치'를 눌러 주세요"
            return
        model = self.model_path()
        if model is None:
            self.status = "모델이 없어요 — 설정에서 모델을 받아 주세요"
            return
        tw.sync_app_files()
        args = [*self.backend_args, "--source", self.source.get(), "--out", self.out_path]
        if "--model" not in self.backend_args:
            args += ["--model", model]
        args += ["--quant", read_settings().get("quant", "auto")]
        self.status = f"모델 불러오는 중… ({os.path.basename(model)})"
        self.backend = start_backend(self.args.python or tw.engine_python(), args, self.ui_q, self.gen)

    def model_path(self):
        """설정에서 고른 모델 폴더. 없으면 받아 둔 첫 모델. 하나도 없으면 None"""
        if "--model" in self.backend_args:
            return self.backend_args[self.backend_args.index("--model") + 1]
        want = read_settings().get("model")
        models = tw.installed_models()
        if want and any(p == want for _, p in models):
            return want
        return models[0][1] if models else None

    def restart_backend(self, why):
        self.stop_backend()
        self.status = why
        self.level = 0.0
        self.start_backend()

    def stop_backend(self):
        if self.backend is not None and self.backend.poll() is None:
            self.backend.terminate()
        self.backend = None

    def on_source(self):
        write_setting("source", self.source.get())
        label = dict(SOURCES)[self.source.get()]
        self.stop_backend()
        self.status = f"입력을 '{label}'(으)로 바꾸는 중… 모델 다시 불러오는 중"
        self.level = 0.0
        self.start_backend()  # 같은 파일(--out)에 이어 쓴다

    # ================================================================ 메시지 처리
    def apply(self, gen, m):
        if gen is not None and gen != self.gen:
            return  # 이미 바꾼 백엔드에서 늦게 온 것
        kind = m.get("type")
        if kind == "text":
            src, body = split_source(m.get("text", ""))
            src = m.get("source") or src
            start, end = m.get("start"), m.get("end")
            if start is not None and end is not None and self.tts.is_echo(body, start, end):
                self.echo_dropped += 1
                return
            self.last_infer = m.get("infer_sec", self.last_infer)
            if self.paused:
                return
            if self.auto_read.get():
                self.tts.speak(body)
            self.add_text(m.get("clock"), src, body)
        elif kind == "level":
            self.level = 0.0 if self.paused else float(m.get("rms", 0))
        elif kind == "status":
            self.status = m.get("msg", "")
        elif kind == "error":
            self.add_item({"ts": None, "src": None, "body": "⚠ " + m.get("msg", ""), "done": True, "err": True})
        elif kind == "exited":
            self.add_item({"ts": None, "src": None, "done": True, "err": True,
                           "body": f"⚠ 백엔드가 종료됐습니다 (코드 {m.get('code')}, 터미널 로그 확인)"})
        elif kind == "tts_ready":
            self.voice = f"음성: {m['msg']}" if m["ok"] else m["msg"]
        elif kind == "update":
            self.update_info = m["info"]
            self.update_btn.config(text=f"새 버전 {m['info']['version']}")
            self.update_btn.pack(side="right", padx=4)
        elif kind == "call":  # 다른 스레드에서 화면을 바꿀 때
            m["fn"]()

    def add_item(self, it):
        self.items.append(it)
        del self.items[:-MAX_ITEMS]
        self.dirty = True

    def add_text(self, ts, src, body):
        """문장 단위로 나눠 넣는다. 같은 출처의 마지막 문장이 덜 끝났으면 거기에 이어 붙인다."""
        for seg, done in split_sentences(body):
            last = next((it for it in reversed(self.items) if not it["err"] and it["src"] == src), None)
            if last is not None and not last["done"]:
                last["body"] = f"{last['body'].rstrip()} {seg}"
                words = last["body"].split()
                last["done"] = done or is_sentence_end(words[-1], len(words)) or len(words) >= 22
            else:
                self.add_item({"ts": ts, "src": src, "body": seg, "err": False,
                               "done": done or len(seg.split()) >= 22})
        self.dirty = True

    # ================================================================ 그리기
    def render(self):
        """지금 보기(live/list/detail)를 다시 그린다. 끌어서 고르는 중이면 미룬다."""
        if self.mouse_down:
            return
        self.dirty = False
        t = self.text
        first = t.yview()[0]
        t.config(state="normal")
        t.delete("1.0", "end")
        self.line_items = []
        if self.view == "live":
            for i, it in enumerate(self.items):
                self.insert_line(it["ts"], it["src"], it["body"],
                                 "err" if it["err"] else ("" if it["done"] else "open"))
                self.line_items.append(None if it["err"] else i)
            if not self.items:
                t.insert("end", "말하는 순간, 바로 텍스트로 — 실시간 전사 대기 중…\n", "partial")
                self.line_items.append(None)
            if self.paused:
                t.insert("end", "일시정지됨 — 재개를 누르면 다시 실시간 전사를 시작합니다\n", "partial")
        elif self.view == "list":
            if not self.history:
                t.insert("end", f"아직 저장된 녹음이 없습니다. ({TALKIT_DOCS})\n", "partial")
            for i, e in enumerate(self.history):
                t.insert("end", e["title"], "htitle")
                t.insert("end", f"   {e['count']}문장   ", "ts")
                t.insert("end", e["preview"] + "\n")
                self.line_items.append(i)
        else:
            entry, lines = self.detail
            if not lines:
                t.insert("end", "이 녹음에는 전사된 문장이 없습니다.\n", "partial")
            for i, (clock, src, body) in enumerate(lines):
                self.insert_line(clock, src, body, "")
                self.line_items.append(i)
        t.config(state="disabled")
        if self.view == "live" and self.follow.get():
            t.see("end")
        else:
            t.yview_moveto(first)

    def insert_line(self, ts, src, body, tag):
        t = self.text
        if ts:
            t.insert("end", f"{ts}  ", "ts")
        if src:
            t.insert("end", f" {src} ", src)
            t.insert("end", " ")
        t.insert("end", body + "\n", tag or ())

    def refresh_status(self):
        el = int(time.time() - self.started)
        if self.paused:
            rec, color = "❚❚ 일시정지됨", MUTED
        else:
            rec, color = f"● 녹음 중 {el // 60:02d}:{el % 60:02d}", RED
        for w in (self.rec, self.mini_rec):
            w.config(text=rec, fg=color)
        label = "재개" if self.paused else "일시정지"
        self.pause_btn.config(text=label)
        self.mini_pause.config(text=label)
        for w in (self.wave, self.mini_wave):
            w.push(self.level)
            w.draw(self.level > 0.08 and not self.paused)
        self.level *= 0.6  # 소리 크기 메시지가 끊기면 막대가 서서히 내려가게

        parts = [self.voice]
        if self.last_infer is not None:
            parts.append(f"전사 {self.last_infer:.2f}초")
        if self.echo_dropped:
            parts.append(f"TTS 소리라서 건너뜀 {self.echo_dropped}개")
        parts.append(f"저장: {self.out_path}")
        if self.toast and time.time() < self.toast_until:
            parts.insert(0, self.toast)
        self.info.config(text=" · ".join(parts) + "\n" + self.status)
        if self.mini:
            self.mini_text.config(text=self.tail(110) or "말하는 순간, 바로 텍스트로.")

    def tail(self, n):
        full = " ".join(it["body"] for it in self.items if not it["err"])
        if len(full) <= n:
            return full
        cut = full[-n:]
        sp = cut.find(" ", 0, 12)
        return "… " + (cut[sp + 1:] if sp >= 0 else cut)

    def poll(self):
        try:
            while True:
                self.apply(*self.ui_q.get_nowait())
        except queue.Empty:
            pass
        if self.dirty and self.view == "live":
            self.render()
        self.refresh_status()
        self.root.after(100, self.poll)

    # ================================================================ 마우스
    def line_at(self, event):
        ix = int(self.text.index(f"@{event.x},{event.y}").split(".")[0]) - 1
        return self.line_items[ix] if 0 <= ix < len(self.line_items) else None

    def on_motion(self, event):
        t = self.text
        t.tag_remove("hover", "1.0", "end")
        ix = int(t.index(f"@{event.x},{event.y}").split(".")[0])
        if 0 < ix <= len(self.line_items) and self.line_items[ix - 1] is not None:
            t.tag_add("hover", f"{ix}.0", f"{ix}.end")

    def on_release(self, event):
        self.mouse_down = False
        if self.text.tag_ranges("sel"):  # 끌어서 골랐으면 복사
            self.copy_selection()
        else:
            ix = self.line_at(event)
            if ix is not None:
                if self.view == "list":
                    self.open_entry(ix)
                elif self.view == "live":
                    self.read(self.items[ix]["body"])
                else:
                    self.read(self.detail[1][ix][2])
        if self.dirty and self.view == "live":
            self.render()

    def read(self, body):
        self.tts.stop()
        self.tts.speak(body)

    # ================================================================ 복사·비우기
    def to_clipboard(self, text, what):
        if not text.strip():
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.toast, self.toast_until = what, time.time() + 2.5

    def copy_selection(self):
        """고른 범위에서 시각·출처 표시는 빼고 문장만 복사한다."""
        t = self.text
        if not t.tag_ranges("sel"):
            return
        skip = {"ts", "시스템", "마이크"}
        active = set(t.tag_names("sel.first")) & skip  # 고른 곳이 시각 한가운데서 시작했을 때
        parts = []
        for key, value, _ in t.dump("sel.first", "sel.last", text=True, tag=True):
            if key == "tagon" and value in skip:
                active.add(value)
            elif key == "tagoff":
                active.discard(value)
            elif key == "text" and not active:
                parts.append(value)
        lines = [l.strip() for l in "".join(parts).splitlines()]
        self.to_clipboard("\n".join(l for l in lines if l), "선택한 글 복사됨")

    def copy_all(self):
        if self.view == "detail":
            lines = [f"[{s}] {b}" if s else b for _, s, b in self.detail[1]]
        else:
            lines = [f"[{it['src']}] {it['body']}" if it["src"] else it["body"]
                     for it in self.items if not it["err"]]
        self.to_clipboard("\n".join(lines), f"전체복사됨 ({len(lines)}문장)")

    def clear(self):
        self.items.clear()
        self.dirty = True
        self.render()

    # ================================================================ 일시정지·따라가기·항상 위
    def toggle_pause(self):
        self.paused = not self.paused
        self.level = 0.0
        self.dirty = True
        self.render()

    def on_follow(self):
        write_setting("follow", int(self.follow.get()))
        if self.follow.get() and self.view == "live":
            self.text.see("end")

    def on_topmost(self):
        write_setting("topmost", int(self.topmost.get()))
        self.root.attributes("-topmost", self.topmost.get() or self.mini)

    # ================================================================ 기록
    def open_history(self):
        self.history = scan_history(os.path.dirname(os.path.abspath(self.out_path)))
        self.view, self.detail = "list", None
        self.show_hbar(f"녹음 기록 {len(self.history)}개", refresh=True)
        self.text.yview_moveto(0)
        self.render()
        self.text.yview_moveto(0)

    def open_entry(self, ix):
        e = self.history[ix]
        self.detail = (e, read_history(e["path"]))
        self.view = "detail"
        self.show_hbar(f"{e['title']} · {len(self.detail[1])}문장", detail=True)
        self.render()
        self.text.yview_moveto(0)

    def close_history(self):
        self.view, self.detail = "live", None
        self.hbar.pack_forget()
        self.dirty = True
        self.render()

    def show_hbar(self, title, refresh=False, detail=False):
        for w in (self.hbar_list, self.hbar_refresh, self.hbar_copy):
            w.pack_forget()
        if detail:
            self.hbar_list.pack(side="left", padx=4)
            self.hbar_copy.pack(side="right")
        if refresh:
            self.hbar_refresh.pack(side="right")
        self.htitle.config(text=title)
        self.htitle.pack_forget()
        self.htitle.pack(side="left", padx=10)
        self.hbar.pack(fill="x", padx=14, pady=(0, 6), before=self.text.master)

    # ================================================================ 미니창
    def toggle_mini(self):
        self.mini = not self.mini
        if self.mini:
            self.studio_geom = self.root.geometry()
            self.studio.pack_forget()
            self.minif.pack(fill="both", expand=True)
            self.root.geometry("520x150")
            self.root.attributes("-topmost", True)
        else:
            self.minif.pack_forget()
            self.studio.pack(fill="both", expand=True)
            if self.studio_geom:
                self.root.geometry(self.studio_geom)
            self.root.attributes("-topmost", self.topmost.get())
            self.dirty = True

    # ================================================================ 업데이트
    def bg_check_update(self):
        if read_settings().get("auto_update", "1") != "1":
            return
        try:
            info = tw.check_update()
        except Exception:
            return  # 인터넷이 없거나 GitHub 이 안 될 때는 조용히 넘어간다
        if info:
            self.ui_q.put((None, {"type": "update", "info": info}))

    # ================================================================ 설정 창
    def open_settings(self):
        if self.settings_win is not None and self.settings_win.winfo_exists():
            self.settings_win.lift()
            return
        self.settings_win = SettingsWindow(self)

    def close(self):
        self.stop_backend()
        self.tts.close()
        self.root.destroy()


class SettingsWindow:
    """엔진 설치 · 모델 받기/고르기 · 양자화 · 업데이트. 오래 걸리는 일은 스레드에서 하고 진행 상황을 아래에 보여 준다."""

    def __init__(self, app):
        self.app = app
        self.busy = False
        w = self.win = tk.Toplevel(app.root)
        w.title("Talkit 설정")
        w.geometry("620x700")
        w.configure(bg=WHITE)
        w.attributes("-topmost", app.root.attributes("-topmost"))
        self.body = tk.Frame(w, bg=WHITE)
        self.body.pack(fill="x", padx=16, pady=(12, 4))
        self.prog = ttk.Progressbar(w, mode="determinate", maximum=100)
        self.prog.pack(fill="x", padx=16, pady=(8, 2))
        self.log = tk.Text(w, height=9, font=("Consolas", 8), bg="#f7f8fc", fg=BLACK, bd=0, wrap="char",
                           state="disabled")
        self.log.pack(fill="both", expand=True, padx=16, pady=(2, 12))
        self.draw()

    # ---- 그리기
    def section(self, title):
        tk.Label(self.body, text=title, font=(FONT, 11, "bold"), fg=BLUE, bg=WHITE, anchor="w"
                 ).pack(fill="x", pady=(10, 2))

    def row(self):
        f = tk.Frame(self.body, bg=WHITE)
        f.pack(fill="x", pady=1)
        return f

    def label(self, parent, text, muted=False, side="left"):
        tk.Label(parent, text=text, font=(FONT, 9), fg=MUTED if muted else BLACK, bg=WHITE, anchor="w",
                 justify="left").pack(side=side)

    def draw(self):
        for c in self.body.winfo_children():
            c.destroy()
        st = read_settings()
        gpu = tw.nvidia_gpu()
        ready = tw.engine_ready()

        self.section("전사 엔진")
        r = self.row()
        if tw.FROZEN:
            self.label(r, "설치됨 ✅" if ready else "아직 설치 안 됨 — 처음 한 번 필요 (Python + torch + qwen-asr, 몇 GB)")
            if not ready:
                pill(r, "엔진 설치", self.do_install, primary=True).pack(side="right")
        else:
            self.label(r, f"소스 실행: 지금 Python 사용 ({sys.executable})", muted=True)
        r = self.row()
        self.label(r, f"GPU: {gpu[0]} ({round(gpu[1] / 1024)}GB)" if gpu else "GPU: NVIDIA GPU 없음 → CPU 로 전사 (느림)",
                   muted=True)

        self.section("모델")
        models = tw.installed_models()
        cur = self.app.model_path()
        self.model_var = tk.StringVar(value=cur or "")
        if not models:
            self.label(self.row(), "받아 둔 모델이 없어요. 아래에서 하나 받아 주세요.", muted=True)
        for name, path in models:
            r = self.row()
            tk.Radiobutton(r, text=f"{name}   {tw.dir_size(path) / 1e9:.1f}GB", value=path, variable=self.model_var,
                           command=self.on_model, bg=WHITE, activebackground=WHITE, font=(FONT, 9)).pack(side="left")
        have = {os.path.basename(p) for _, p in models}
        for m in tw.MODELS:
            r = self.row()
            self.label(r, f"• {m['label']}  {m['gb']}GB", muted=True)
            if m["folder"] in have:
                self.label(r, "받음", muted=True, side="right")
            else:
                pill(r, "받기", lambda m=m: self.do_download(m["repo"], m["folder"])).pack(side="right")
        r = self.row()
        pill(r, "다른 모델 받기 (Hugging Face 주소)", self.ask_download).pack(side="left")

        self.section("양자화 (불러올 때의 정밀도)")
        self.quant_var = tk.StringVar(value=st.get("quant", "auto"))
        for val, text in tw.QUANTS:
            if val == "auto":
                text += f" → 지금 PC: {dict(tw.QUANTS)[tw.auto_quant()].split(' (')[0]}"
            tk.Radiobutton(self.row(), text=text, value=val, variable=self.quant_var, command=self.on_quant,
                           bg=WHITE, activebackground=WHITE, font=(FONT, 9),
                           state="normal" if (gpu or val not in ("int8", "nf4")) else "disabled").pack(side="left")
        r = self.row()
        self.label(r, "고른 모델을 양자화해서 저장 (GPU 필요, 다음부터 빨리 불러옴):", muted=True)
        pill(r, "8-bit 저장", lambda: self.do_quantize(8)).pack(side="right")
        pill(r, "4-bit 저장", lambda: self.do_quantize(4)).pack(side="right", padx=4)

        self.section("업데이트")
        r = self.row()
        self.label(r, f"지금 버전: {tw.APP_VERSION}" + ("" if tw.FROZEN else " (소스 실행 — git pull 로 업데이트)"))
        info = self.app.update_info
        if info:
            pill(r, f"{info['version']} 로 업데이트", self.do_update, primary=True).pack(side="right")
            pill(r, "바뀐 점", lambda: webbrowser.open(info["page"])).pack(side="right", padx=4)
        else:
            pill(r, "지금 확인", self.do_check).pack(side="right")
        self.auto_upd = tk.BooleanVar(value=st.get("auto_update", "1") == "1")
        check(self.row(), "켤 때 새 버전 확인", self.auto_upd,
              lambda: write_setting("auto_update", int(self.auto_upd.get()))).pack(side="left")

    # ---- 진행 상황 (다른 스레드에서 불러도 됨)
    def say(self, msg):
        def f():
            if not self.win.winfo_exists():
                return
            self.log.config(state="normal")
            self.log.insert("end", msg + "\n")
            self.log.see("end")
            self.log.config(state="disabled")
            m = re.search(r"(\d+)%", msg)
            if m:
                self.prog.config(value=int(m.group(1)))
        self.app.ui_q.put((None, {"type": "call", "fn": f}))

    def progress(self, done, total):
        def f():
            if self.win.winfo_exists():
                self.prog.config(value=done * 100 / total if total else 0)
        self.app.ui_q.put((None, {"type": "call", "fn": f}))

    def job(self, name, fn, after=None):
        if self.busy:
            messagebox.showinfo("Talkit", "다른 작업이 끝날 때까지 기다려 주세요.", parent=self.win)
            return
        self.busy = True
        self.prog.config(value=0)
        self.say(f"▶ {name}")

        def run():
            ok = True
            try:
                fn()
            except Exception as e:
                ok = False
                self.say(f"✗ {name} 실패: {e}")
            finally:
                self.busy = False

            def end():
                if self.win.winfo_exists():
                    self.draw()
                if ok and after:
                    after()
            self.app.ui_q.put((None, {"type": "call", "fn": end}))
        threading.Thread(target=run, daemon=True).start()

    def tool(self, args):
        """model_tool.py 실행. error 가 오면 예외"""
        err = []

        def on(m):
            k = m.get("type")
            if k == "progress":
                self.progress(m["done"], m["total"])
                self.say(f"{m['done'] / 1e9:.2f} / {m['total'] / 1e9:.2f} GB") if int(time.time()) % 5 == 0 else None
            elif k == "error":
                err.append(m["msg"])
            elif k == "done":
                self.say("완료: " + ", ".join(f"{a}={b}" for a, b in m.items() if a != "type"))
            else:
                self.say(m.get("msg", str(m)))
        code = tw.run_model_tool(args, on)
        if err or code != 0:
            raise RuntimeError(err[0] if err else f"종료 코드 {code}")

    # ---- 동작
    def need_engine(self):
        if tw.engine_ready():
            return False
        messagebox.showinfo("Talkit", "먼저 '엔진 설치'를 해 주세요.", parent=self.win)
        return True

    def do_install(self):
        if not messagebox.askokcancel(
                "엔진 설치", "전사 엔진을 설치합니다.\n\nPython·torch·qwen-asr 를 내려받아 %LOCALAPPDATA%\\Talkit 에 둡니다."
                "\nGPU 가 있으면 약 4GB, 없으면 약 1GB, 10~30분 걸립니다.", parent=self.win):
            return
        self.job("엔진 설치", lambda: tw.install_engine(self.say), after=self.after_change)

    def do_download(self, repo, folder):
        if self.need_engine():
            return
        dst = os.path.join(tw.MODELS_DIR, folder)
        self.job(f"{repo} 받기", lambda: self.tool(["download", "--repo", repo, "--dir", dst]),
                 after=lambda: self.pick_if_none(dst))

    def ask_download(self):
        repo = simpledialog.askstring("다른 모델 받기", "Hugging Face 모델 주소 (예: Qwen/Qwen3-ASR-0.6B)",
                                      parent=self.win)
        if repo and re.fullmatch(r"[\w.-]+/[\w.-]+", repo.strip()):
            self.do_download(repo.strip(), repo.strip().split("/")[1])
        elif repo:
            messagebox.showwarning("Talkit", "'이름/모델' 모양으로 적어 주세요.", parent=self.win)

    def do_quantize(self, bits):
        if self.need_engine():
            return
        src = self.app.model_path()
        if not src:
            messagebox.showinfo("Talkit", "먼저 모델을 받아 주세요.", parent=self.win)
            return
        if not tw.nvidia_gpu():
            messagebox.showinfo("Talkit", "양자화해서 저장하기는 NVIDIA GPU 가 필요해요.\n"
                                "GPU 가 없으면 위에서 'CPU 8-bit' 를 고르세요 (불러올 때마다 양자화).", parent=self.win)
            return
        name = os.path.basename(src.rstrip("\\/"))
        dst = os.path.join(tw.MODELS_DIR, f"{name}-{'nf4' if bits == 4 else 'int8'}")
        if os.path.exists(dst):
            messagebox.showinfo("Talkit", f"이미 있어요: {dst}", parent=self.win)
            return
        if not messagebox.askokcancel("양자화", f"{name} 을 {bits}-bit 로 양자화해 저장합니다.\n"
                                      "GPU 메모리를 비우려고 전사를 잠깐 멈춥니다.", parent=self.win):
            return
        self.app.stop_backend()
        self.app.status = "양자화하는 동안 전사를 멈췄어요"

        def done():
            write_setting("model", dst)
            self.app.restart_backend(f"양자화한 모델로 다시 시작… ({os.path.basename(dst)})")
        self.job(f"{bits}-bit 양자화", lambda: self.tool(["quantize", "--src", src, "--dst", dst, "--bits", str(bits)]),
                 after=done)

    def do_check(self):
        def f():
            info = tw.check_update()
            self.app.update_info = info
            self.say(f"새 버전 {info['version']} 이 있어요" if info else f"최신 버전이에요 ({tw.APP_VERSION})")
            if info:
                self.app.ui_q.put((None, {"type": "update", "info": info}))
        self.job("업데이트 확인", f)

    def do_update(self):
        info = self.app.update_info
        if not tw.FROZEN:
            webbrowser.open(info["page"])
            return

        def after():
            self.app.close()
        self.job(f"{info['version']} 받기", lambda: tw.apply_update(info, self.say), after=after)

    def on_model(self):
        write_setting("model", self.model_var.get())
        self.app.restart_backend(f"모델을 바꾸는 중… ({os.path.basename(self.model_var.get())})")

    def on_quant(self):
        write_setting("quant", self.quant_var.get())
        self.app.restart_backend(f"정밀도를 '{dict(tw.QUANTS)[self.quant_var.get()]}'(으)로 바꾸는 중…")

    def pick_if_none(self, path):
        if not read_settings().get("model") or self.app.backend is None:
            write_setting("model", path)
        self.after_change()

    def after_change(self):
        if self.app.backend is None:
            self.app.start_backend()


def main():
    p = argparse.ArgumentParser(description="Talkit for Windows (backend.py + Windows 기본 음성)")
    p.add_argument("--python", default=None,
                   help="backend.py 를 돌릴 Python (기본: exe 판은 설치한 엔진, 소스 실행은 지금 Python)")
    p.add_argument("--source", choices=[s for s, _ in SOURCES], default=None, help="처음 입력 (기본: 지난번 설정)")
    p.add_argument("--out", default=None, help="저장할 txt (기본: 문서\\Talkit\\transcript_날짜_시각.txt)")
    p.add_argument("--topmost", action="store_true", help="창을 항상 위에")
    p.add_argument("--auto-read", action="store_true", help="처음부터 자동 읽기 켜기")
    args, backend_args = p.parse_known_args()

    try:  # 고해상도 화면에서 글자가 흐리지 않게
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root, args, backend_args)
    root.mainloop()


if __name__ == "__main__":
    main()
