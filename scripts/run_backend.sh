#!/usr/bin/env bash
# WSL2 안에서 전사 백엔드를 실행한다. 추가 옵션은 그대로 넘어간다.
#   bash scripts/run_backend.sh             # 4초 단위
#   bash scripts/run_backend.sh --chunk 3   # 3초 단위
#   MODEL=/다른/경로 bash scripts/run_backend.sh
set -euo pipefail

# shellcheck disable=SC1091
. "$HOME/.cargo/env"
cd "$(dirname "$0")/../backend"

MODEL="${MODEL:-$HOME/models/qwen3_asr_1.7b_ko}"
# Windows 폴더(/mnt/c/...)에서 빌드하면 느리므로 결과물은 Linux 쪽에 둔다
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/asr-backend-target}"

exec cargo run --release -- \
  --model "$MODEL" \
  --python "$HOME/.venvs/asr/bin/python" \
  --worker asr_worker.py \
  "$@"
