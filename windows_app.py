"""
Windows 전사 화면 (tkinter) — backend.py 를 실행해 전사 결과를 보여 주고, Windows 기본 음성으로 읽어 준다.

Rust 앱(app/)을 빌드할 수 없는 PC(스마트 앱 컨트롤이 켜진 Windows)용이다. 추가 설치가 필요 없다.

- 백엔드: `python -X utf8 backend.py [옵션]` 의 stdout JSON 을 한 줄씩 읽는다 (약속은 README 참고).
- TTS: PowerShell 의 System.Speech(SAPI)를 하위 프로세스로 띄워 한국어 음성(Microsoft Heami)으로 읽는다.
  "읽기 멈춤"은 그 프로세스를 끄고 새로 띄운다.
- 메아리 거르기: TTS 가 읽던 시간과 겹치고 내용도 비슷한 전사는 TTS 소리가 다시 잡힌 것으로 보고 버린다
  (app/src/tts.rs 의 is_echo 와 같은 규칙).

실행
    python windows_app.py                       # 마이크 + 시스템 소리
    python windows_app.py --source system --topmost
    python windows_app.py --input-wav a.wav     # 녹음 대신 파일로 시험
그 밖의 옵션(--chunk, --context, --pause 등)은 그대로 backend.py 로 넘어간다.
"""

import argparse
import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL = os.path.normpath(os.path.join(HERE, "..", "qwen3_asr_1.7b_ko"))
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
ECHO_MARGIN = 1.0  # 메아리 판단 때 앞뒤 여유(초)


# ---------------------------------------------------------------- 메아리 판단
def similarity(a: str, b: str) -> float:
    """두 문장이 얼마나 비슷한가 (0~1, 공백·기호 무시 글자 2개 묶음 기준)"""

    def bigrams(s):
        cs = [c for c in s if c.isalnum()]
        return [(cs[i], cs[i + 1]) for i in range(len(cs) - 1)]

    x, y = bigrams(a), bigrams(b)
    if not x or not y:
        return 0.0
    total = len(x) + len(y)
    common = 0
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
def start_backend(python, backend_args, ui_q):
    cmd = [python, "-X", "utf8", os.path.join(HERE, "backend.py"), *backend_args]
    try:
        proc = subprocess.Popen(
            cmd, cwd=HERE, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=None,
            creationflags=NO_WINDOW if python.lower().endswith("pythonw.exe") else 0,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
    except OSError as e:
        ui_q.put({"type": "error", "msg": f"백엔드를 실행할 수 없습니다: {e}"})
        return None

    def read():
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ui_q.put(json.loads(line))
            except json.JSONDecodeError:
                ui_q.put({"type": "status", "msg": line})
        ui_q.put({"type": "exited", "code": proc.wait()})

    threading.Thread(target=read, daemon=True).start()
    return proc


# ---------------------------------------------------------------- 창
class App:
    def __init__(self, root, args, backend_args):
        self.root = root
        self.ui_q = queue.Queue()
        self.lines = []  # (본문, 오류인가)
        self.echo_dropped = 0
        self.last_infer = None
        self.voice = "음성 준비 중…"
        self.status = "백엔드 시작 중…"

        root.title("실시간 전사 (Windows)")
        root.geometry("760x520")
        root.attributes("-topmost", args.topmost)

        bar = tk.Frame(root)
        bar.pack(fill="x", padx=8, pady=(8, 4))
        self.auto_read = tk.BooleanVar(value=args.auto_read)
        tk.Checkbutton(bar, text="자동 읽기", variable=self.auto_read).pack(side="left")
        tk.Button(bar, text="읽기 멈춤", command=lambda: self.tts.stop()).pack(side="left", padx=4)
        self.topmost = tk.BooleanVar(value=args.topmost)
        tk.Checkbutton(bar, text="항상 위", variable=self.topmost,
                       command=lambda: root.attributes("-topmost", self.topmost.get())).pack(side="left")
        tk.Label(bar, text="문장을 누르면 읽어 줘요", fg="gray").pack(side="right")

        self.text = tk.Text(root, wrap="word", font=("Malgun Gothic", 13), cursor="hand2",
                            state="disabled", spacing1=3, spacing3=3)
        self.text.pack(fill="both", expand=True, padx=8)
        self.text.tag_configure("error", foreground="#c62828")
        self.text.tag_configure("hover", background="#e8eef7")
        self.text.bind("<Button-1>", self.on_click)
        self.text.bind("<Motion>", self.on_motion)

        self.info = tk.Label(root, anchor="w", fg="gray", justify="left")
        self.info.pack(fill="x", padx=8, pady=(4, 8))

        self.tts = Tts(lambda ok, msg: self.ui_q.put({"type": "tts_ready", "ok": ok, "msg": msg}))
        self.backend = start_backend(args.python, backend_args, self.ui_q)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh_info()
        self.poll()

    # ---- 화면 갱신
    def refresh_info(self):
        parts = [self.voice]
        if self.last_infer is not None:
            parts.append(f"전사 {self.last_infer:.2f}초")
        if self.echo_dropped:
            parts.append(f"TTS 소리라서 건너뜀 {self.echo_dropped}개")
        self.info.config(text=" · ".join(parts) + "\n" + self.status)

    def push_line(self, shown, body, is_error=False):
        self.lines.append((body, is_error))
        self.text.config(state="normal")
        self.text.insert("end", shown + "\n", ("error",) if is_error else ())
        self.text.config(state="disabled")
        self.text.see("end")

    def line_at(self, event):
        ix = int(self.text.index(f"@{event.x},{event.y}").split(".")[0]) - 1
        return ix if 0 <= ix < len(self.lines) else None

    def on_click(self, event):
        ix = self.line_at(event)
        if ix is not None and not self.lines[ix][1]:
            self.tts.stop()
            self.tts.speak(self.lines[ix][0])
        return "break"  # 글자 선택 대신 읽기

    def on_motion(self, event):
        self.text.tag_remove("hover", "1.0", "end")
        ix = self.line_at(event)
        if ix is not None and not self.lines[ix][1]:
            self.text.tag_add("hover", f"{ix + 1}.0", f"{ix + 1}.end")

    # ---- 백엔드·TTS 메시지 처리
    def apply(self, m):
        kind = m.get("type")
        if kind == "text":
            text, start, end = m.get("text", ""), m.get("start"), m.get("end")
            if start is not None and end is not None and self.tts.is_echo(text, start, end):
                self.echo_dropped += 1
                return
            self.last_infer = m.get("infer_sec", self.last_infer)
            if self.auto_read.get():
                self.tts.speak(text)
            clock = m.get("clock")
            self.push_line(f"[{clock}] {text}" if clock else text, text)
        elif kind == "status":
            self.status = m.get("msg", "")
        elif kind == "error":
            msg = m.get("msg", "")
            self.push_line(f"⚠ {msg}", msg, is_error=True)
        elif kind == "exited":
            self.push_line(f"⚠ 백엔드가 종료됐습니다 (코드 {m.get('code')}, 터미널 로그 확인)", "", True)
        elif kind == "tts_ready":
            self.voice = f"음성: {m['msg']}" if m["ok"] else m["msg"]

    def poll(self):
        try:
            while True:
                self.apply(self.ui_q.get_nowait())
        except queue.Empty:
            pass
        self.refresh_info()
        self.root.after(100, self.poll)

    def close(self):
        if self.backend is not None and self.backend.poll() is None:
            self.backend.terminate()
        self.tts.close()
        self.root.destroy()


def main():
    p = argparse.ArgumentParser(description="Windows 전사 화면 (backend.py + Windows 기본 음성)")
    p.add_argument("--python", default=os.environ.get("RECORDER_PYTHON", sys.executable),
                   help="backend.py 를 돌릴 Python (torch·qwen-asr 설치된 것)")
    p.add_argument("--topmost", action="store_true", help="창을 항상 위에")
    p.add_argument("--auto-read", action="store_true", help="처음부터 자동 읽기 켜기")
    args, backend_args = p.parse_known_args()

    # 모델을 따로 안 정했으면 옆 폴더(../qwen3_asr_1.7b_ko)에 받아 둔 모델을 쓴다
    if "--model" not in backend_args and os.path.isdir(LOCAL_MODEL):
        backend_args += ["--model", LOCAL_MODEL]

    root = tk.Tk()
    App(root, args, backend_args)
    root.mainloop()


if __name__ == "__main__":
    main()
