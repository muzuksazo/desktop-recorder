"""연습용 가짜 백엔드.

진짜 backend.py 대신, 같은 형식의 JSON 한 줄씩을 몇 초마다 출력한다.
모델이나 마이크는 전혀 쓰지 않아서, 화면(디자인)만 빠르게 확인할 때 쓴다.

화면에서 쓰기:  $env:QWEN_BACKEND="fake"; cargo run
혼자 실행해 보기:  python fake_backend.py   (끝내려면 Ctrl+C)
"""

import datetime as dt
import json
import time

SENTENCES = [
    "안녕하세요, 오늘 수업을 시작하겠습니다.",
    "지난 시간에 배운 내용을 먼저 복습해 볼게요.",
    "화면에 보이는 코드를 같이 따라 쳐 보세요.",
    "질문 있으면 채팅으로 남겨 주세요.",
    "이 부분이 오늘 시험에 나올 수도 있어요.",
    "잠깐 쉬었다가 10분 뒤에 다시 시작하겠습니다.",
]


def send(**msg):
    """한 줄 = JSON 하나, 바로 flush."""
    print(json.dumps(msg, ensure_ascii=False), flush=True)


def main():
    send(type="status", msg="모델 로딩 중… (가짜)")
    time.sleep(1.5)
    send(type="status", msg="마이크: 가짜 마이크")
    send(type="status", msg="듣는 중… (가짜 백엔드)")

    i = 0
    while True:
        time.sleep(1.5)
        if i == 3:
            send(type="error", msg="연습용 오류 메시지입니다 (무시해도 됨)")
        now = time.time()
        clock = dt.datetime.fromtimestamp(now).strftime("%H:%M:%S")
        send(type="text", text=SENTENCES[i % len(SENTENCES)], clock=clock,
             start=round(now - 1.5, 3), end=round(now, 3), infer_sec=0.1)
        i += 1


if __name__ == "__main__":
    main()
