#!/usr/bin/env bash
set -euo pipefail

automl-nas search --config "${1:-configs/cifar10.yaml}"
