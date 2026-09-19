# Multi-Trial Neural Architecture Search Prototype

This repository is a small, local neural architecture search prototype for
CIFAR-10. It uses PyTorch to instantiate discrete CNN candidates and Ray Tune
to orchestrate parallel trials. Optuna TPE or seeded random sampling proposes
architectures, while ASHA can stop weak trials early.

The project is an experimentation prototype. It is not a production platform,
a demonstrated multi-node distributed system, a cloud cost optimizer, or a
weight-sharing/one-shot NAS implementation.

## Implemented workflow

```text
CIFAR-10 training data
  -> deterministic train/validation split
  -> conditional 2-4 block CNN search space
  -> PyTorch candidate training
  -> validation metrics reported to Ray Tune
  -> ASHA early stopping
  -> best candidate selected by validation accuracy
```

The official CIFAR-10 test set is not used during architecture search. It is
reserved for a later final-evaluation phase.

## Search space

[`config/default.yaml`](config/default.yaml) is the executable source of truth.
For each active block, the search selects:

- kernel size: 3 or 5;
- output channels: 16, 32, or 64;
- activation: ReLU or SiLU;
- residual connection: enabled or disabled.

Only parameters for active blocks are sampled. Training hyperparameters are
fixed in this phase so architecture comparisons are not mixed with a separate
hyperparameter search.

## Supported environment

Phase 1 targets the following explicit environment:

- Python 3.11.9
- PyTorch 2.4.0
- torchvision 0.19.0
- Ray Tune 2.40.0
- Optuna 4.0.0
- PyYAML 6.0.2

Install dependencies in a Python 3.11 virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run a CPU search

```bash
python -m src.orchestrator.main --config config/default.yaml --cpu
```

The launcher forwards the same CLI options:

```bash
./scripts/run_search.sh --cpu
```

For a cheap end-to-end smoke run:

```bash
python -m src.orchestrator.main \
  --config config/default.yaml \
  --strategy random \
  --trials 2 \
  --parallelism 1 \
  --max-epochs 1 \
  --max-train-samples 128 \
  --max-validation-samples 64 \
  --cpu
```

The first run downloads CIFAR-10 to `data/`. Ray output is written under
`ray_results/`. Both locations are ignored by Git.

## GPU mode

GPU mode is optional:

```bash
python -m src.orchestrator.main --config config/default.yaml --gpu
```

GPU trials explicitly request one GPU through Ray. On a one-GPU machine, Ray
therefore schedules at most one GPU trial at a time and does not silently
oversubscribe the device.

## Checkpoints

Each reported epoch registers a Ray checkpoint containing model state,
optimizer state, completed epoch, trial configuration, Python/PyTorch RNG
state, and DataLoader generator state. This supports meaningful trial resume
for the selected Ray version. Full cross-environment deterministic recovery is
not claimed.

## Current limitations

- No final benchmark or performance claim is included yet.
- No official test-set evaluation is performed during search.
- Only local Ray execution has been validated.
- The current search is multi-trial NAS, not a weight-sharing SuperNet.
- Dependency locking, a full test suite, CI, and the final experiment protocol
  belong to later phases.
- ASHA is described only as resource-efficient trial pruning; no monetary cost
  reduction is claimed or measured.

The old sample metrics file was removed because it was illustrative and could
not be reproduced from the executable search path.

## License

MIT. See [`LICENSE`](LICENSE).
