#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python main.py \
  --data_name Beauty \
  --eval_only \
  --checkpoint_path ./output/RADRec-Beauty.pt \
  "$@"
