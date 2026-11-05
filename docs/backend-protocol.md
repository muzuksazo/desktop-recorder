# 화면(Rust, ui/) ↔ 전사(Python, backend.py) 연결 규칙

> `sangyun` 브랜치의 `backend.py` 약속을 그대로 따른다. (backend.py 맨 위 설명과 같음)

## 큰 그림

```
[ui/ Rust·GPUI 화면] ──실행──▶ [backend.py]
        ▲                        - 마이크·시스템 소리 녹음, Silero VAD로 문장 단위 자르기
        │                        - Qwen3-ASR 전사, txt 저장
        └── stdout JSON 한 줄씩 ◀┘
```

## 실행

- 화면이 `<python> -X utf8 -u backend.py [옵션]`을 실행한다.
- `cargo run -- 옵션`처럼 뒤에 붙인 옵션은 backend.py에 그대로 넘어간다.
  - 예: `--source mic|system|both`, `--model 경로`, `--context "GPUI, 러스트"`, `--out 경로.txt`
- `<python>`: 환경변수 `QWEN_PYTHON`(또는 `RECORDER_PYTHON`), 없으면 `python`
- 연습용: `QWEN_BACKEND=fake`면 `ui/fake_backend.py`를 실행한다 (모델 없이 화면만 확인).

## 출력 형식 (stdout, UTF-8, 한 줄 = JSON 하나)

| type | 필드 | 예시 |
|---|---|---|
| `status` | `msg` | `{"type":"status","msg":"듣는 중… (device=cuda:0, …)"}` |
| `text` | `text`, `clock`, `start`, `end`, `infer_sec` | `{"type":"text","text":"오늘 수업은…","clock":"14:31:05","start":1730790665.2,"end":1730790668.9,"infer_sec":0.84}` |
| `error` | `msg` | `{"type":"error","msg":"모델 로딩 실패: …"}` |

- `start`/`end`: 그 소리가 녹음된 시각(Unix 초). `infer_sec`: 전사에 걸린 시간.
- 로그·경고는 stderr로만 나간다 (화면을 실행한 터미널에 보인다).
- 화면은 모르는 type·필드를 무시한다.
- 화면은 `msg`가 "듣는 중"으로 시작하는 status를 받거나 첫 `text`를 받으면 상태 점을 초록으로 바꾼다.

## 종료

- 화면은 backend.py의 **stdin을 쓰지 않는다** (Windows에서 stdin을 기다리면 torch import가 멈춘다).
- 창을 닫으면 화면이 백엔드 프로세스를 끈다. txt는 문장마다 바로 저장되므로 안전하다.

## 모델

- Windows(이 PC: RTX 4070 8GB): `TeamUNIVA/qwen3_asr_1.7b_ko` 원본 (CUDA)
- Mac(Apple Silicon)에서 쓴다면: MLX 4bit 변환본 `yeowon/qwen3_asr_1.7b_ko-MLX-4bit` 검토 (backend.py 수정 필요)
