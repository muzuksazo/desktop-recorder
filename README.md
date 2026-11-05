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
