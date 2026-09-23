#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "benchmark_status: ENVIRONMENT_MISSING"
  echo "expected_python: $PYTHON"
  exit 2
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$PYTHON" "$ROOT_DIR/scripts/final_benchmark.py" status
