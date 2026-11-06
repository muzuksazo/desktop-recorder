#!/usr/bin/env bash
# WSL2(Ubuntu) 안에서 한 번만 실행: Rust + 전사 워커용 Python 환경을 설치한다.
#   bash scripts/setup_wsl.sh [Windows 쪽 모델 폴더]
# 예) bash scripts/setup_wsl.sh /mnt/c/Users/user/Desktop/ne4/qwen3_asr_1.7b_ko
set -euo pipefail

VENV="$HOME/.venvs/asr"
MODEL_SRC="${1:-}"
MODEL_DST="$HOME/models/qwen3_asr_1.7b_ko"

echo "== 1/4 시스템 패키지"
sudo apt-get update
sudo apt-get install -y build-essential pkg-config curl python3-venv python3-dev libsndfile1

echo "== 2/4 Rust"
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
# shellcheck disable=SC1091
. "$HOME/.cargo/env"
cargo --version

echo "== 3/4 Python 워커 환경 ($VENV)"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -U pip
"$VENV/bin/pip" install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/pip" install qwen-asr soundfile numpy
"$VENV/bin/python" -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available())"

echo "== 4/4 모델"
if [ -n "$MODEL_SRC" ]; then
  # /mnt/c 에서 바로 읽으면 느리므로 Linux 쪽으로 복사해 둔다 (약 4.7GB)
  mkdir -p "$(dirname "$MODEL_DST")"
  cp -r "$MODEL_SRC" "$MODEL_DST"
  echo "모델 복사 완료: $MODEL_DST"
else
  echo "모델 폴더를 안 줬습니다. run_backend.sh 실행 때 MODEL=경로 로 지정하세요."
fi

echo "끝. 다음: bash scripts/run_backend.sh"
