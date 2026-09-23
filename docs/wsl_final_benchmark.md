# WSL final benchmark runbook

This runbook covers the frozen **pre-test** benchmark only. The Windows partial final-search
runs are `ABORTED — EnvironmentMigrationToWSL`; never copy their Ray state or results here.
The protocol, scheduler calibration, and comparative pilot remain valid.

## 1. WSL requirements

- Ubuntu under WSL 2 with the Windows NVIDIA driver exposed through `nvidia-smi`.
- NVIDIA GeForce RTX 4060 Laptop GPU with approximately 8 GB VRAM.
- Git, `tmux`, and VS Code's WSL support.
- Do not install a Linux NVIDIA display driver in WSL.

Verify the host-facing GPU and tmux:

```bash
nvidia-smi
tmux -V
```

If tmux is missing, install it manually with:

```bash
sudo apt update && sudo apt install -y tmux
```

## 2. Repository location

Run only from the native Linux filesystem:

```bash
cd /home/amana/projects/AutoML-System-with-Neural-Architecture-Search
git switch final-benchmark
git status --short --branch
```

Do not run the benchmark from `/mnt/c` and do not copy a Windows checkout or Windows Ray state.

## 3. Python environment

The validated environment uses Python 3.11.9 in `.venv`. If it must be recreated without
changing Ubuntu's system Python:

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/automl-uv-install.sh
sh /tmp/automl-uv-install.sh
$HOME/.local/bin/uv python install 3.11.9
$HOME/.local/bin/uv venv --python 3.11.9 .venv
source .venv/bin/activate
```

## 4. CUDA install

Install the frozen Linux CUDA environment and the project without dependency re-resolution:

```bash
UV_HTTP_TIMEOUT=300 $HOME/.local/bin/uv pip install \
  --python .venv/bin/python \
  --index-strategy unsafe-best-match \
  -r requirements.cuda-linux.lock
$HOME/.local/bin/uv pip install \
  --python .venv/bin/python \
  --no-deps --no-build-isolation -e .
.venv/bin/python -m pip check
```

The multi-index option is required because the lock combines packages from PyPI with PyTorch
CUDA 12.4 wheels. It does not loosen any version pins. The CPU lock remains `requirements.lock`
for CI and ordinary development.

## 5. Preflight

The safe dry-run validates the frozen configs, exact software versions, CUDA, deterministic
training, and a one-GPU Ray worker. It does not run a search or use CIFAR-10:

```bash
./scripts/run_final_benchmark.sh --dry-run
```

`CUBLAS_WORKSPACE_CONFIG=:4096:8` is exported before Python imports PyTorch. Warnings are errors
during the deterministic CUDA smoke step.

## 6. VS Code

From the WSL repository, run:

```bash
code .
```

In VS Code choose `/home/amana/projects/AutoML-System-with-Neural-Architecture-Search/.venv/bin/python`
as the Python interpreter. Do not select a Windows Python executable.

## 7. tmux

Create a durable terminal session:

```bash
tmux new -s automl-final
```

Detach by pressing `Ctrl+B`, then `D`. Reconnect with:

```bash
tmux attach -t automl-final
```

## 8. Starting the benchmark

Run these commands manually inside the tmux session:

```bash
cd /home/amana/projects/AutoML-System-with-Neural-Architecture-Search
source .venv/bin/activate
./scripts/run_final_benchmark.sh
```

The runner stops at `PRETEST_FREEZE_COMPLETE`. It never evaluates the official test split.

## 9. Detaching and reconnecting

Detaching does not stop the process. After reconnecting, leave the runner in its original shell.
Use a second WSL terminal for status and log inspection.

## 10. Status command

```bash
cd /home/amana/projects/AutoML-System-with-Neural-Architecture-Search
source .venv/bin/activate
./scripts/final_benchmark_status.sh
```

The status command is read only. Before the first run it reports `NOT_STARTED`.

## 11. Log inspection

Per-stage logs are under `artifacts/final_runner/logs/`. For example:

```bash
tail -f artifacts/final_runner/logs/random_2026.log
```

Other logs cover each search, validation, aggregation, confirmation, final training, and freeze.

## 12. Recovery after interruption

Run the same command again. Completed stages are skipped only after their artifacts and
provenance validate. An interrupted search resumes through Ray only when its config, Git SHA,
Linux environment, dependency versions, and persisted Ray state all match. Otherwise the run is
recorded as aborted and a clean search is required. Confirmation and final training are accepted
only when their complete summaries and required artifacts validate.

Never copy or reconstruct Ray history manually.

## 13. What invalidates a run

The runner fails immediately for a dirty working tree, changed Git SHA, changed frozen-config
hash, changed dependency/GPU identity, unavailable CUDA, unexpected GPU, failed trial, incomplete
artifact, non-Linux provenance, or an official-test flag that is not false. Windows final-search
artifacts are never eligible. If code or a frozen config must change, preserve the old state for
audit, then start a new benchmark state only after review.

## 14. Official-test prohibition

The orchestration code contains no official-test stage. Do not run the locked-test CLI or open the
CIFAR-10 test partition before the pre-test freeze is reviewed and separately authorized. The
freeze artifact is written to `results/final/pretest_freeze.json` only after all pre-test work is
complete, with `official_test_access` set to `false`.

## 15. What to send back after completion

Do not paste full logs. Send only:

- the output of `./scripts/final_benchmark_status.sh`;
- the path and content of `artifacts/final_runner/final_search_summary.json` and
  `results/final/pretest_freeze.json`;
- any failure message, if present.
