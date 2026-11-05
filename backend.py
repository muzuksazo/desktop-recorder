"""
전사 백엔드 (화면 없음) — GPUI 앱(app/)이 실행하고, stdout 을 한 줄씩 읽어 화면에 표시한다.

stdout 약속 (앱 ↔ 백엔드, UTF-8, 한 줄에 JSON 하나)
    {"type": "status",  "msg": "모델 로딩 중…"}            상태 표시줄용
    {"type": "partial", "text": "실시간 스트리밍 인식 중…"} 진행 중인 실시간 문장
    {"type": "text",    "text": "[14:31:05] 전사된 문장"}   확정된 전사 결과
    {"type": "error",   "msg": "…"}                        오류
로그·경고는 stderr 로만 나간다.
종료: 앱이 stdin 을 닫으면(앱 종료) 백엔드도 끝난다.
"""

import argparse
import datetime as dt
import json
import os
import queue
import sys
import threading


def main():
    # stdout 은 앱과 주고받는 통로라서, 라이브러리가 찍는 print 는 stderr 로 돌린다
    out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def emit(**msg):
        out.write(json.dumps(msg, ensure_ascii=False) + "\n")

    import live_transcribe as lt

    p = argparse.ArgumentParser(description="전사 백엔드 (GPUI 앱용, 화면 없음)")
    p.add_argument("--model", default=lt.MODEL_ID, help="모델 폴더 경로 또는 HF 모델 ID")
    p.add_argument("--source", choices=["both", "mic", "system"], default="both")
    p.add_argument("--chunk", type=float, default=6.0, help="최대 발화 구간 길이(초, 기본 6)")
    p.add_argument("--device", default="auto", help="auto / mlx / cuda:0 / mps / cpu")
    p.add_argument("--system-device", default=None, help="시스템 소리 장치 이름(일부만 써도 됨)")
    p.add_argument("--silence", type=float, default=0.005, help="이 값보다 조용하면 건너뜀")
    p.add_argument("--out", default=None, help="저장할 txt 경로")
    args = p.parse_args()

    if args.out is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.out = os.path.join(here, dt.datetime.now().strftime("transcript_%Y-%m-%d_%H%M.txt"))

    ui_q = queue.Queue()
    stop = threading.Event()

    def wait_parent():
        sys.stdin.read()  # 앱이 끝나면 EOF
        stop.set()

    threading.Thread(target=wait_parent, daemon=True).start()

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
    while not stop.is_set():
        try:
            kind, msg = ui_q.get(timeout=0.2)
        except queue.Empty:
            if not worker.is_alive():  # 모델 로딩 실패 등으로 전사가 멈춤
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


if __name__ == "__main__":
    sys.exit(main())
