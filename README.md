# 실시간 한국어 강의 전사 (1시간 MVP)

ZEP 수업 소리를 마이크로 받아 3~5초 단위로 한국어 전사하고, 터미널에 이어서 보여주며 `.txt`로 자동 저장합니다.

- 모델: [TeamUNIVA/qwen3_asr_1.7b_ko](https://huggingface.co/TeamUNIVA/qwen3_asr_1.7b_ko) (Apache-2.0)
- MVP 대상: Mac(M1 Pro, 16GB) 1대
- 입력: 마이크만 (시스템 소리 캡처는 다음 단계)

> ⚠️ **헤드폰 말고 스피커로 수업을 들어야 합니다.** MVP는 마이크만 듣기 때문에, 헤드폰을 끼면 강사 목소리가 잡히지 않습니다.

---

## 1. 설치

### 1-1. Python 확인

터미널을 열고 아래 명령을 입력합니다.

```bash
python3 --version
```

- `Python 3.10` 이상이면 다음 단계로 넘어갑니다. (최소 버전은 확인 안 됨, 추정값)
- 명령을 찾을 수 없다고 나오면 [python.org](https://www.python.org/downloads/)에서 설치합니다.

### 1-2. 작업 폴더와 가상환경 만들기

```bash
mkdir -p ~/Documents/qwen-live
cd ~/Documents/qwen-live
python3 -m venv .venv
source .venv/bin/activate
```

터미널 앞부분에 `(.venv)`가 보이면 성공입니다. 터미널을 새로 열 때마다 `source .venv/bin/activate`를 다시 입력해야 합니다.

### 1-3. 패키지 설치

```bash
pip install -U pip
pip install -U qwen-asr sounddevice numpy
```

### 1-4. 마이크 권한 (Mac)

처음 실행할 때 "터미널이 마이크에 접근하려고 합니다"라는 창이 뜨면 **허용**을 누릅니다.
창이 안 뜨거나 이미 거부했다면 **시스템 설정 → 개인정보 보호 및 보안 → 마이크**에서 터미널(또는 사용하는 앱)을 켭니다.

---

## 2. 실행

```bash
cd ~/Documents/qwen-live
source .venv/bin/activate
python live_transcribe.py
```

- 첫 실행 때 모델 파일(약 4GB 추정)을 내려받느라 시간이 걸립니다.
- 전사 결과는 터미널에 `[14:31:05] 전사된 문장` 형식으로 계속 이어집니다.
- 같은 내용이 `transcript_날짜_시각.txt` 파일에 한 줄씩 저장됩니다.
- 끝낼 때는 `Ctrl + C`를 누릅니다.

> `live_transcribe.py`는 아직 작성 전입니다. T5·T6에서 만듭니다.

---

## 3. 할 일 (T1~T7, 총 약 60분)

| # | 할 일 | 예상 시간 | 완료 기준 |
|---|---|---|---|
| T1 | 환경 준비: Python 확인, 가상환경 생성 (1-1, 1-2) | 5분 | `(.venv)` 표시 확인 |
| T2 | 패키지 설치 (1-3) | 5분 | 설치 오류 없음 |
| T3 | 모델 다운로드 및 음성 파일 1개 전사 테스트 | 15분 | 한국어 문장이 출력됨 |
| T4 | 장치 확인: `device_map="mps"`로 시도하고 실패하면 `"cpu"`로 전환, 5초 음성의 전사 시간 측정 | 5분 | 전사 시간이 5초 미만인지 확인 |
| T5 | 실시간 루프: 4초 녹음 → 전사 → 터미널 출력 (녹음과 전사를 스레드로 분리) | 20분 | 말한 내용이 몇 초 뒤 계속 나타남 |
| T6 | `.txt` 자동 저장: 구간마다 시각을 붙여 한 줄씩 추가, 빈 결과는 저장 안 함 | 5분 | 종료 후 파일에 내용이 있음 |
| T7 | 실제 수업 테스트: ZEP 수업을 스피커로 틀고 5분 받아쓰기 | 5분 | 지연(초)과 체감 정확도 기록 |

**판단 지점:** T4에서 5초 음성의 전사가 5초를 넘으면 실시간 루프가 밀립니다. 이때는 구간을 6~8초로 늘리거나 더 작은 모델을 검토합니다.

---

## 4. 팀원 역할 분담 (제안)

| 담당 | 할 일 | 결과물 |
|---|---|---|
| Mac A | T1~T6 개발 | `live_transcribe.py` |
| Mac B | T3부터 같이 따라 하며 막힌 지점 기록, T7 테스트 | 막힌 지점 메모, 테스트 결과 (지연·정확도) |
| Windows 1 | Windows 3대의 GPU 사양 조사 (작업 관리자 → 성능 → GPU) | PC별 GPU 이름, 전용 GPU 메모리, RAM 표 |
| Windows 2 | Windows에서 같은 스크립트를 돌릴 준비 (Python 설치, 1-2·1-3 따라 하기) | 설치 성공 여부, 오류 메시지 |
| Windows 3 | 다음 단계 조사: 시스템 소리 캡처 방법 (Mac은 BlackHole, Windows는 WASAPI loopback) | 방법별 설치 절차 요약 |

---

## 5. 다음 단계 (MVP 이후)

- 시스템 소리 캡처 (헤드폰 사용 가능하게)
- Windows 3대 적용
- 터미널 대신 별도 창 UI
- 구간 경계에서 단어가 잘리는 문제 개선

# desktop-recorder

ZEP 수업 소리(시스템 소리)와 마이크 소리를 **TeamUNIVA/qwen3_asr_1.7b_ko** 모델로 받아 적고, 데스크탑 창에 실시간으로 보여 주는 도구입니다.

- 4초 단위로 잘라서 전사합니다. 그래서 글자는 말하고 나서 대략 4~8초 뒤에 뜹니다(PC 속도에 따라 다름, 미검증).
- 전사 결과는 `transcript_날짜_시각.txt`에 한 줄씩 바로 저장됩니다. 중간에 꺼져도 그때까지 내용은 남습니다.

> ⚠ 아직 실제 PC에서 돌려 보지 않은 코드입니다. 처음 실행할 때 오류가 나면 오류 문구를 그대로 공유해 주세요.

## 1. 설치 (처음 한 번)

Python **3.11 또는 3.12**를 권장합니다. (3.13 이상은 torch가 아직 안 맞을 수 있음)

```bash
git clone https://github.com/muzuksazo/desktop-recorder.git
cd desktop-recorder
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

Mac:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows + NVIDIA GPU가 있는 경우
기본 `pip install torch`는 CPU 버전이 깔릴 수 있습니다. GPU를 쓰려면 https://pytorch.org 에서 CUDA 버전 설치 명령을 받아 torch를 다시 설치하세요.
GPU 확인: 작업 관리자 → 성능 → GPU 항목에 NVIDIA가 있는지 확인.

### Mac에서 시스템 소리(ZEP 강사 목소리) 받기
Mac은 기본 기능으로 시스템 소리를 녹음할 수 없어서 **BlackHole 2ch**가 필요합니다.
1. `brew install blackhole-2ch` (또는 https://existential.audio/blackhole 에서 설치)
2. "오디오 MIDI 설정" 앱 → 왼쪽 아래 `+` → **다중 출력 기기** 만들기 → 스피커(또는 이어폰)와 BlackHole 2ch 둘 다 체크
3. 시스템 설정 → 사운드 → 출력을 방금 만든 다중 출력 기기로 변경
4. 처음 실행할 때 터미널의 마이크 권한 요청을 허용

Windows는 추가 설치 없이 기본 스피커 소리를 그대로 받습니다.

## 2. 실행

```bash
python live_transcribe.py
```

처음 실행하면 모델(약 4GB)을 내려받느라 몇 분 걸립니다.

| 옵션 | 설명 |
|---|---|
| `--source mic` / `system` / `both` | 마이크만 / 시스템 소리만 / 둘 다(기본) |
| `--chunk 3` | 3초 단위로 전사 (짧을수록 빨리 뜨지만 문장이 잘림) |
| `--device cpu` | 장치 강제 지정 (`auto`, `cuda:0`, `mps`, `cpu`) |
| `--topmost` | 창을 항상 위에 |
| `--list` | 오디오 장치 이름 목록 보기 |
| `--system-device 이름` | 시스템 소리 장치를 직접 지정 |

## 3. 알려진 한계

- 공식 스트리밍 기능은 vLLM + NVIDIA GPU 전용이라, 이 스크립트는 "짧게 잘라 반복 전사" 방식입니다. 구간 경계에서 단어가 잘릴 수 있습니다.
- Mac(M1 Pro)에서 MPS 가속이 되는지는 확인 안 됨입니다. 느리면 `--device cpu`로 비교해 보세요.
- GPU 없는 Windows에서 1.7B 모델이 4초 안에 전사를 끝낼지는 확인 안 됨입니다. 밀리면 `--chunk`를 늘리세요.

## 4. 팀 테스트 체크리스트

- [ ] Mac 1: 설치·실행, 마이크 전사 확인
- [ ] Mac 1: BlackHole 설정 후 ZEP 소리 전사 확인
- [ ] Windows 1: 실행, GPU 유무와 지연 시간 기록
- [ ] 나머지 PC: 같은 순서로 확인
- [ ] txt 파일 저장 확인

---

## GPUI 앱 + Python 백엔드 (sangyun 브랜치)

```
[app/ Rust·GPUI 화면]  ──실행──▶  [backend.py Python]
        ▲                          - 마이크·시스템 소리 녹음, Qwen3-ASR 전사 (live_transcribe.py 재사용)
        └──── stdout JSON 한 줄씩 ──┘
```

### 백엔드 ↔ 앱 약속 (백엔드 팀과 맞출 부분)
- 앱이 `python -X utf8 backend.py [옵션]`을 실행한다. 앱 실행 옵션은 그대로 백엔드에 넘어간다.
- 백엔드는 stdout에 UTF-8 JSON을 한 줄씩 쓴다. 로그는 stderr로만 쓴다.
  - `{"type": "status", "msg": "…"}` 상태
  - `{"type": "text", "text": "[14:31:05] 전사된 문장"}` 전사 결과
  - `{"type": "error", "msg": "…"}` 오류
- `text` 줄에는 `clock`(표시용 시각), `start`/`end`(녹음 시각, Unix 초), `infer_sec`(전사 시간)도 들어간다.
  앱은 `start`/`end`가 TTS 재생 구간과 겹치면 TTS 소리가 다시 잡힌 것으로 보고 버린다.
- 앱 창을 닫으면 앱이 백엔드 프로세스를 끈다.
- 주의(Windows): 백엔드에서 stdin을 읽으며 기다리면 torch import가 멈춘다. stdin은 쓰지 않는다.

### 실행 (Windows)
1. Visual Studio Build Tools(C++를 사용한 데스크톱 개발)와 Rust(rustup) 설치
2. torch·qwen-asr·soundcard가 설치된 Python을 지정하고 실행:
   ```powershell
   $env:RECORDER_PYTHON = "C:\경로\.venv\Scripts\python.exe"
   cd app
   cargo run -- --model C:\경로\qwen3_asr_1.7b_ko --source both
   ```
   - `--source mic|system|both`, `--chunk 4`, `--out 경로.txt` 등은 `live_transcribe.py`와 같다.
   - 첫 빌드는 4분쯤 걸린다.

### 전사 방식 (속도·정확도)
- **VAD 문장 단위**: Silero VAD로 말이 0.6초 끊기면 바로 전사한다. 고정 4초 단위보다 빨리 뜨고 단어 중간에서 안 잘린다.
  15초 넘게 이어지면 마지막 3초 중 가장 조용한 곳에서 자른다.
- 말 시작 전 0.3초를 같이 넣는다. 이 모델은 앞에 무음이 0.5초 이상 붙으면 첫 단어를 빼먹는다(시험 결과).
- 언어는 한국어로 고정한다. 앞 문장은 문맥으로 넘기지 않는다(시험해 보니 더 틀림).
  수업 용어는 `--context "GPUI, 러스트, 백엔드"`로 넘기면 도움이 된다.
- Windows에서는 백엔드의 효율 모드(EcoQoS)를 끈다. 창 없는 프로세스라 CPU가 느려져 전사가 2배 느렸다.
- 조절 옵션: `--pause 0.6`(문장 끝으로 볼 무음 초), `--max-seg 15`, `--vad 0.5`
- 테스트: `--input-wav a.wav b.wav`로 녹음 대신 파일을 실시간 속도로 넣을 수 있다.

### TTS (전사 결과 읽기)
- Windows 기본 음성(SAPI)을 쓴다. 한국어 음성(Microsoft Heami)이 있으면 자동으로 고른다. 오프라인, 추가 설치 없음.
- 문장을 누르면 읽는다. "자동 읽기"를 켜면 새 문장마다 읽는다. "읽기 멈춤"으로 끊는다.
- 시스템 소리를 전사 중이면 TTS 소리도 녹음된다. 읽는 동안 들어온 소리는 버리므로, 그동안 나온 수업 내용은 빠질 수 있다.
