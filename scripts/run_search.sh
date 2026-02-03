#!/bin/bash

# Enterprise NAS Launcher
# Usage: ./run_search.sh [bayesian|random] [trials]

STRATEGY=${1:-bayesian}
TRIALS=${2:-10}

echo "==========================================="
echo "   Enterprise NAS System - Launcher"
echo "==========================================="
echo "Strategy: $STRATEGY"
echo "Trials:   $TRIALS"
echo "==========================================="

# Ensure PYTHONPATH is set so that src module is found
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the Python Entry Point (Updated path)
python src/orchestrator/main.py --strategy $STRATEGY --trials $TRIALS --parallelism 2 --gpu

echo "==========================================="
echo "   Search Complete. Check ./results"
echo "==========================================="
