> 📢 **Talkit for Windows 공개** — Mac 앱 Talkit 1.2.0의 Windows 판입니다. [Releases](https://github.com/muzuksazo/desktop-recorder/releases)에서 `Talkit.exe`를 받아 실행하세요 (모델 받기·양자화·자동 업데이트 포함).
> 설치·기능은 아래 [Talkit for Windows](#talkit-for-windows-windows_apppy) 절을 보세요.

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

## Windows + WSL2 구조 (sangyub 브랜치)

스마트 앱 컨트롤이 켜진 Windows에서는 새로 빌드한 Rust 실행 파일과 일부 Python DLL이 막힙니다. 그래서 다음처럼 나눕니다.

```
[Windows] gui_client.py (tkinter)          [WSL2] backend/ (Rust)
 - 마이크·시스템 소리 녹음      ──오디오──▶  - 4초씩 모으기, 무음 건너뛰기
 - 전사 결과 표시·txt 저장      ◀──텍스트──  - asr_worker.py (GPU, Qwen3-ASR) 호출
                localhost:7700 (TCP)
```

### 처음 한 번
1. 관리자 PowerShell에서 `wsl --install` 실행 후 재부팅, Ubuntu 사용자 만들기
2. WSL(Ubuntu) 터미널에서:
   ```bash
   cd /mnt/c/Users/<사용자>/Desktop/ne4/desktop-recorder
   bash scripts/setup_wsl.sh /mnt/c/Users/<사용자>/Desktop/ne4/qwen3_asr_1.7b_ko
   ```
3. Windows Python에 `pip install numpy soundcard soundfile` (torch 필요 없음)

### 실행
1. WSL: `bash scripts/run_backend.sh` → "GUI 접속 대기 중"이 나올 때까지 기다림 (첫 빌드는 몇 분)
2. Windows: `python gui_client.py` (`--source system`, `--topmost` 등 사용 가능)

---

## Talkit Mac 앱 (DMG 설치 · 모델 위치)

### 설치

1. `Talkit-버전.dmg`를 열고 **Talkit**을 **응용 프로그램(Applications)** 폴더로 끌어다 놓습니다.
2. 처음 열면 **"Apple은 'Talkit'에 악성 코드가 없음을 확인할 수 없습니다"** 창이 뜹니다. 아직 Apple 공증을 받지 않은 앱이라서 그렇습니다. 아래 둘 중 하나로 엽니다.
   - **방법 A (설정 화면)**: 경고 창에서 **완료** → 시스템 설정 → **개인정보 보호 및 보안** → 아래로 내려 "'Talkit'이(가) 차단되었습니다" 옆 **그래도 열기** → Mac 암호 입력 → 다시 뜨는 창에서 **열기**.
     (macOS 15 이상은 우클릭 → 열기로는 안 넘어갑니다.)
   - **방법 B (터미널 한 줄)**: 터미널에 붙여 넣고 실행한 뒤 Talkit을 엽니다.
     ```bash
     xattr -dr com.apple.quarantine /Applications/Talkit.app
     ```
3. 녹음 권한을 허용합니다. (마이크 / 시스템 오디오 녹음)
   나중에 바꾸려면: 시스템 설정 → 개인정보 보호 및 보안 → 화면 및 시스템 오디오 녹음 → **시스템 오디오 녹음만** 에서 Talkit 켜기.

필요 조건: Apple Silicon(M1 이상), macOS 14.4 이상, 첫 실행 때 인터넷과 여유 공간 약 1.2GB.
Python은 앱 안에 들어 있어서 따로 설치할 필요가 없습니다.

### 모델 위치

모델은 DMG에 들어 있지 않습니다. 처음 실행하면 자동으로 내려받습니다(약 1.1GB, 화면에 % 표시).

| 무엇 | 위치 |
|---|---|
| 음성 인식 모델 | `~/Library/Application Support/Talkit/models/qwen3_asr_1.7b_ko-MLX-4bit/` |
| 설정 | `~/Library/Application Support/Talkit/settings.txt` |
| 전사 기록 | `~/Documents/Talkit/transcript_YYYY-MM-DD_HHMM.txt` |

- 모델 원본: Hugging Face [`yeowon/qwen3_asr_1.7b_ko-MLX-4bit`](https://huggingface.co/yeowon/qwen3_asr_1.7b_ko-MLX-4bit) (공개)
- 인터넷이 안 되는 Mac에는 다른 Mac에서 받아 위 폴더에 그대로 넣으면 됩니다. 폴더 안에 `model.safetensors`, `config.json` 등이 바로 있어야 합니다.
  ```bash
  hf download yeowon/qwen3_asr_1.7b_ko-MLX-4bit --local-dir "$HOME/Library/Application Support/Talkit/models/qwen3_asr_1.7b_ko-MLX-4bit"
  ```
- 모델 폴더를 다른 곳에 두려면 환경 변수 `TALKIT_MODELS_DIR`로 바꿀 수 있습니다.

### DMG 만들기 (개발자)

```bash
packaging/package_macos.sh                      # 기본 버전
VERSION=1.2.1 BUILD=8 packaging/package_macos.sh
```

- 외부 배포판(DMG)은 `edition.txt = public` 이 들어가 **공개 4-bit 모델만** 씁니다.
- 비공개 엔진은 소스에서 실행하는 개발용 앱에서만 쓰며, 저장소와 DMG 어디에도 들어가지 않습니다.

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

---

## Talkit for Windows (`windows_app.py`)

Mac 앱 **Talkit 1.2.0**의 Windows 판입니다. ZEP 강의 소리(시스템 소리)와 마이크를 실시간으로 받아 적습니다.
Rust 빌드가 필요 없어서 스마트 앱 컨트롤이 켜진 Windows에서도 돌아갑니다(`app/` 빌드는 os error 4551로 차단됨). WSL도 필요 없습니다.

```
[windows_app.py 화면(tkinter)]  ──실행──▶  [backend.py]  (GPUI 앱과 같은 stdout JSON 약속)
        └── TTS: PowerShell System.Speech (Microsoft Heami)
```

### exe로 쓰기 (권장, Python 설치 필요 없음)
1. [Releases](https://github.com/muzuksazo/desktop-recorder/releases)에서 `win-v…` 릴리스의 **`Talkit.exe`**(약 13MB)를 받아 실행합니다.
   - "Windows의 PC 보호" 창이 뜨면 **추가 정보 → 실행** (서명 인증서가 없어서 뜹니다).
2. 처음 켜면 설정 창이 열립니다. **엔진 설치**를 누릅니다 (Python 3.14.7 + torch + qwen-asr 를 `%LOCALAPPDATA%\Talkit\runtime`에 설치. GPU 있으면 약 4GB, 없으면 약 1GB, 10~30분).
   - NVIDIA GPU가 있으면 CUDA 판, 없으면 CPU 판 torch를 저절로 고릅니다.
3. **모델 받기**: 한국어 1.7B(기본) / Qwen 공식 1.7B / Qwen 공식 0.6B(가벼움) 중에서 받습니다. `다른 모델 받기`로 Hugging Face 주소를 적어 받을 수도 있습니다.
4. 받으면 바로 전사가 시작됩니다. 다음부터는 exe만 켜면 됩니다.

| 위치 | 내용 |
|---|---|
| `%LOCALAPPDATA%\Talkit\runtime` | 전사 엔진 (Python + 패키지) |
| `%LOCALAPPDATA%\Talkit\models` | 받은 모델, 양자화해서 저장한 모델 |
| `%LOCALAPPDATA%\Talkit\app` | exe 안의 엔진 스크립트 (버전이 바뀔 때마다 새로 풀림) |
| `문서\Talkit` | 전사 기록 txt |
| `%APPDATA%\Talkit\settings.txt` | 설정 (입력, 모델, 양자화, 따라가기 …) |

### 양자화 (설정 → 양자화)
모델을 불러올 때의 정밀도를 고릅니다. **자동**은 GPU 6GB 이상이면 원본, 그보다 작으면 4-bit, GPU가 없으면 CPU 8-bit.
"고른 모델을 4-bit로 저장"을 누르면 양자화한 모델을 따로 저장해 두고 다음부터 그걸 바로 불러옵니다.

측정 (2026-11-06, RTX 4070 Laptop 8GB / 같은 노트북 CPU, 한국어 12초 음성 한 번에 전사, 결과 글자는 모두 같음):

| 방식 | 메모리 | 전사 시간 | 불러오기 |
|---|---|---|---|
| GPU 원본 bf16 | 4.2GB (GPU) | 8.2초 | 23초 |
| GPU 8-bit | 2.5GB (GPU) | 43.1초 ⚠ 느림 | 39초 |
| GPU 4-bit | 1.6GB (GPU) | 7.3초 | 28초 |
| **GPU 4-bit 저장본** (4.7GB → 1.5GB) | 1.6GB (GPU) | 3.5초 | **9초** |
| CPU 원본 fp32 | 5.2GB (RAM) | 35.3초 | 15초 |
| **CPU 8-bit** | 6.8GB (RAM, 불러올 때 최고치) | **7.4초** | 41초 |

- GPU 없는 PC는 **CPU 8-bit**가 원본보다 4.8배 빠릅니다(12초 음성을 7초에 → 실시간으로 따라감). 더 느린 CPU면 0.6B 모델을 쓰세요.
- 4-bit/8-bit 저장본은 bitsandbytes 형식이라 NVIDIA GPU에서만 불러옵니다.

### 자동 업데이트
- 켤 때 GitHub 릴리스에서 `win-v…` 태그의 새 `Talkit.exe`가 있는지 확인합니다 (설정에서 끌 수 있음). 있으면 위에 **새 버전** 버튼이 뜹니다.
- 업데이트를 누르면 새 exe를 받아 지금 exe 옆에 두고, 앱이 꺼지면 바꿔 끼운 뒤 다시 켭니다. 엔진이 쓰는 패키지 목록이 바뀐 버전이면 다시 '엔진 설치'를 안내합니다.

### exe 만들기·배포 (개발자)
```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1     # → dist\Talkit.exe
gh release create win-v1.0.0 dist\Talkit.exe --title "Talkit for Windows 1.0.0"
```
- 버전은 `talkit_win.py`의 `APP_VERSION`. 올린 뒤 태그를 `win-v<버전>`으로, 첨부 파일 이름을 `Talkit.exe`로 해야 자동 업데이트가 찾습니다.
- 엔진 패키지는 `packaging/requirements-windows.txt`에 고정 (torch·bitsandbytes는 GPU 유무에 따라 따로 설치).

### 소스에서 실행 (개발용)
1. Python 3.11 이상, `pip install -r requirements.txt` (GPU면 CUDA 판 torch, 양자화하려면 `bitsandbytes`)
2. 모델: `../qwen3_asr_1.7b_ko` 폴더에 받아 두면 그걸 쓰고, 설정 창에서 받을 수도 있습니다.

```powershell
python windows_app.py                        # 지난번 입력 설정으로 (처음엔 마이크 + 시스템 소리)
python windows_app.py --source system --topmost --auto-read
python windows_app.py --input-wav a.wav      # 녹음 대신 파일로 시험
```
- 그 밖의 옵션(`--chunk`, `--context "러스트, 백엔드"`, `--pause` 등)은 그대로 `backend.py`로 넘어갑니다.
- 다른 Python으로 백엔드를 돌리려면 `--python 경로` 또는 환경 변수 `RECORDER_PYTHON`.

### 기능 (Mac 1.2.0과 비교)

| 기능 | Windows |
|---|---|
| 녹음 소스 선택: 마이크 / 시스템 / 둘 다 | ✅ 바꾸면 같은 파일에 이어 씀 |
| 시스템 / 마이크 출처 표시 | ✅ 구간마다 더 크게 들린 쪽 |
| 전사 따라가기 켜고 끄기 | ✅ |
| 이전 녹음 기록 목록·다시 보기 | ✅ `문서\Talkit\transcript_*.txt` |
| 미니창(PiP) | ✅ 항상 위, 최근 전사 이어 보기 (잠금은 아직) |
| 일시정지·재개, 녹음 시간, 소리 크기 막대 | ✅ |
| 문장 단위로 나눠 보기 | ✅ |
| 끌어서 복사, 전체복사, 비우기 | ✅ 끌어서 복사하면 시각·출처 없이 문장만 |
| TTS (전사 결과 읽기) | ✅ Windows 판에만 있음 |
| 설치 파일 | ✅ `Talkit.exe` (엔진·모델은 처음 실행 때 앱 안에서 설치) |
| 모델 받기 · 양자화 · 자동 업데이트 | ✅ Windows 판에만 있음 (아래 참고) |

- 설정은 `%APPDATA%\Talkit\settings.txt`에 기억합니다 (입력, 따라가기, 자동 읽기, 항상 위).
- 강의 소리 파일은 저장하지 않습니다. 전사된 글만 남습니다 (Mac 판과 같음).

### TTS
- 문장을 누르면 읽습니다. "자동 읽기"를 켜면 새 문장마다 읽고, "읽기 멈춤"으로 끊습니다.
- 시스템 소리를 받는 중이면 TTS 소리도 녹음됩니다. 읽던 시간과 겹치고 내용이 비슷한(0.5 이상) 전사는 TTS 소리로 보고 버립니다 (`app/src/tts.rs`와 같은 규칙). 그동안 나온 강의 내용은 빠질 수 있습니다.

### 백엔드 변경 (Windows / VAD 경로)
- 마이크와 시스템 소리는 지금처럼 섞어서 한 번만 전사합니다 (스피커 소리가 마이크로 다시 잡혀 같은 말이 두 번 나오는 걸 막기 위해).
- 대신 구간마다 어느 입력이 더 컸는지 보고 Mac 경로와 같은 모양으로 출처를 붙입니다: `{"type": "text", "text": "[시스템] 문장", "source": "시스템", …}`. txt에는 `[14:31:05] [시스템] 문장`.
- 소리 크기 `{"type": "level", "rms": 0~1}`을 0.1초마다 보냅니다.

### 확인한 것 (2026-11-06, Windows 11 + RTX 4070 Laptop)
- 시험 wav 3문장 → 3문장 전사 (문장당 1.0~2.0초, cuda:0), txt 저장, 자동 읽기
- 출처 판단: 가짜 두 입력(한쪽 음성, 한쪽 잡음)으로 6구간 모두 맞음
- 읽기 멈춤, 끌어서 복사(시각·출처 빠짐) 정상
- 실제 ZEP 강의 소리로 긴 시간 시험, 기록 화면·미니창·입력 전환은 직접 써 보며 확인 중

> 참고: 위 "Windows + WSL2 구조"의 `gui_client.py`는 `live_transcribe.py`에서 tkinter 화면이 빠지면서 지금은 실행되지 않습니다. Windows에서는 `windows_app.py`를 쓰세요.
