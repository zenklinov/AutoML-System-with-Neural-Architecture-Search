#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing WSL virtual environment: $PYTHON" >&2
  echo "Follow docs/wsl_final_benchmark.md to create it." >&2
  exit 2
fi

case "${1:-}" in
  "")
    exec "$PYTHON" "$ROOT_DIR/scripts/final_benchmark.py" run
    ;;
  --dry-run)
    exec "$PYTHON" "$ROOT_DIR/scripts/final_benchmark.py" dry-run
    ;;
  *)
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
    ;;
esac
