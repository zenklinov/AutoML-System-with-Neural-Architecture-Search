#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Additional CLI overrides are forwarded.
python -m src.orchestrator.main --config config/default.yaml "$@"
