"""
실시간 한국어 전사 — Windows GUI (전사는 WSL2 의 Rust 백엔드가 한다)

동작 방식
- 마이크와 시스템 소리를 녹음해 섞고, 0.5초마다 백엔드(TCP)로 보낸다.
- 백엔드가 몇 초씩 모아 GPU 로 전사한 결과를 받아 창에 붙이고 txt 로 저장한다.
- 이 PC 에는 torch 가 없어도 된다 (numpy, soundcard, soundfile 만 필요).

실행 예  (먼저 WSL2 에서 scripts/run_backend.sh 실행)
    python gui_client.py                  # 마이크 + 시스템 소리
    python gui_client.py --source system  # 시스템 소리만
    python gui_client.py --list           # 오디오 장치 목록
"""

import argparse
import datetime as dt
import json
import os
import socket
import struct
import sys
import threading
import time
import tkinter as tk

from live_transcribe import App, list_devices, start_recorders


class RemoteTranscriber(threading.Thread):
    """녹음한 소리를 백엔드로 보내고, 받은 전사 결과를 창에 넘긴다."""

    def __init__(self, args, ui_q, stop_event):
        super().__init__(daemon=True)
        self.args = args
        self.ui_q = ui_q
        self.stop_event = stop_event

    def connect(self):
        addr = (self.args.host, self.args.port)
        self.ui_q.put(("status", f"백엔드 연결 중… {addr[0]}:{addr[1]} (WSL2 에서 run_backend.sh 실행)"))
        while not self.stop_event.is_set():
            try:
                sock = socket.create_connection(addr, timeout=3)
                sock.settimeout(None)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return sock
            except OSError:
                time.sleep(1)  # 백엔드가 아직 모델을 올리는 중일 수 있다
        return None

    def run(self):
        sock = self.connect()
        if sock is None:
            return
        threading.Thread(target=self.receive, args=(sock,), daemon=True).start()
        queues = start_recorders(self.args, self.ui_q, self.stop_event)
        try:
            while not self.stop_event.is_set():
                mixed = None
                for _, q in queues:
                    item = q.get()
                    if isinstance(item, Exception):
                        self.ui_q.put(("error", str(item)))
                        return
                    mixed = item if mixed is None else mixed + item
                if len(queues) > 1:
                    mixed = mixed / len(queues)
                payload = mixed.astype("<f4").tobytes()
                sock.sendall(struct.pack("<I", len(payload)) + payload)
        except OSError as e:
            self.ui_q.put(("error", f"백엔드 연결이 끊겼습니다: {e}"))
        finally:
            sock.close()

    def receive(self, sock):
        try:
            for line in sock.makefile("r", encoding="utf-8"):
                msg = json.loads(line)
                kind = msg.get("type")
                if kind == "text":
                    stamp = dt.datetime.now().strftime("%H:%M:%S")
                    text = f"[{stamp}] {msg['text']}"
                    with open(self.args.out, "a", encoding="utf-8") as f:
                        f.write(text + "\n")
                    self.ui_q.put(("text", text))
                elif kind == "status":
                    self.ui_q.put(("status_add", msg["msg"]))
                elif kind == "error":
                    self.ui_q.put(("error", msg["msg"]))
        except (OSError, ValueError) as e:
            if not self.stop_event.is_set():
                self.ui_q.put(("error", f"백엔드 응답 오류: {e}"))


def main():
    p = argparse.ArgumentParser(description="실시간 한국어 전사 GUI (WSL2 Rust 백엔드 사용)")
    p.add_argument("--source", choices=["both", "mic", "system"], default="both",
                   help="녹음할 소리 (기본: both)")
    p.add_argument("--system-device", default=None, help="시스템 소리 장치 이름(일부만 써도 됨)")
    p.add_argument("--host", default="127.0.0.1", help="백엔드 주소 (기본 127.0.0.1)")
    p.add_argument("--port", type=int, default=7700, help="백엔드 포트 (기본 7700)")
    p.add_argument("--topmost", action="store_true", help="창을 항상 위에 띄우기")
    p.add_argument("--out", default=None, help="저장할 txt 경로")
    p.add_argument("--list", action="store_true", help="오디오 장치 목록만 출력")
    args = p.parse_args()

    if args.list:
        list_devices()
        return

    if args.out is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.out = os.path.join(here, dt.datetime.now().strftime("transcript_%Y-%m-%d_%H%M.txt"))

    root = tk.Tk()
    App(root, args, worker_cls=RemoteTranscriber)
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
