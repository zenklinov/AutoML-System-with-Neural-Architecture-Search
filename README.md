# Multi-Trial Neural Architecture Search Prototype

This is a small, reproducible neural architecture search prototype for
CIFAR-10. It instantiates discrete PyTorch CNN candidates and uses Ray Tune,
Optuna, and ASHA for local parallel trial orchestration and early stopping.

The project is not a production platform, a demonstrated multi-node system, or
a weight-sharing NAS implementation. No benchmark or model-quality claim is
included yet. The approved, still-unrun contract is documented in
`docs/experiment_protocol.md`.

## Supported environment

- Python 3.11.9
- PyTorch 2.4.0
- torchvision 0.19.0
- Ray Tune 2.40.0
- Optuna 4.0.0

`pyproject.toml` is the source of truth for direct runtime and development
dependencies. `requirements.lock` records the fully resolved reference CPU
environment used for local validation. It is a practical environment lock, not
a claim of bit-for-bit equivalence across operating systems or CUDA hardware.

## Clean installation

CPU installation:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

To reproduce the locally resolved CPU dependency set, install
`requirements.lock`, then install the package without dependency resolution:

```bash
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps --no-build-isolation
```

CUDA experiments use a separate lock so CPU development and CI remain unchanged:

```bash
python -m venv .venv-cuda
source .venv-cuda/bin/activate  # Windows: .venv-cuda\Scripts\Activate.ps1
python -m pip install -r requirements.cuda.lock
python -m pip install -e . --no-deps --no-build-isolation
python scripts/gpu_smoke.py
```

The CUDA lock retains PyTorch 2.4.0 and torchvision 0.19.0 and selects their
CUDA 12.4 wheels. GPU determinism is requested but is not guaranteed across
different drivers, devices, or CUDA libraries. The CUDA environment is for
experiments; `requirements.lock` remains the CPU CI/development reference.

## Package structure

```text
src/automl_nas/
├── config.py       validated YAML contract
├── data.py         deterministic train/validation data isolation
├── models.py       configurable block and discrete CNN candidate
├── training.py     training, evaluation, checkpoint, and RNG state
├── search.py       Optuna/ASHA/Ray orchestration
├── artifacts.py    provenance manifest and canonical result schema
├── protocol.py     fixed baseline and architecture identity
├── workflows.py    calibration, confirmation, and final training
├── locked_test.py  isolated final test evaluator
└── cli.py          installed command-line interface
```

## Configuration

- `configs/smoke.yaml`: two-trial CPU pipeline check using deterministic
  synthetic data. Its accuracy is not experimental evidence.
- `configs/cifar10.yaml`: approved protocol with a provisional, explicitly
  unlocked final budget.
- `configs/asha_calibration_panel.yaml`: fixed architecture panel for
  full-curve ASHA calibration.

Configurations reject unknown fields and unsupported architecture choices.
The YAML file is authoritative; the CLI selects a config but does not silently
override its values.

Validate a config:

```bash
automl-nas validate-config --config configs/smoke.yaml
```

## Run

CPU smoke search:

```bash
automl-nas search --config configs/smoke.yaml
```

Normal CIFAR-10 configuration:

```bash
automl-nas search --config configs/cifar10.yaml
```

The official CIFAR-10 test partition is not loaded during architecture search.
Candidates use the fixed stratified 45,000/5,000 training-only split. See the
protocol document for stage commands and leakage controls.

## Generated outputs

Each run creates:

```text
artifacts/runs/<run-id>/
├── manifest.json
├── summaries/
│   └── trials.json
└── ray/
    └── tune/                 raw Ray state and checkpoints
```

The manifest records configuration, seeds, software versions, Git state,
platform details, resources, and execution status. The canonical trial summary
contains only training and validation information; it intentionally has no
official test metric.

Generated `artifacts/`, CIFAR data, and checkpoints are ignored by Git. The
`results/` directory is reserved for small curated outputs from a future,
approved experiment protocol.

## Tests and lint

```bash
ruff check .
pytest -m "not smoke"
pytest -m smoke
```

The smoke test uses synthetic data while exercising the same model, training,
Ray reporting, checkpoint, scheduler, search, manifest, and result-export path.
It does not download CIFAR-10 or require a GPU.

## Reproducibility boundaries

- Python, NumPy, PyTorch, dataset-split, DataLoader, and Optuna seeds are
  controlled and recorded.
- Deterministic PyTorch algorithms are requested by the supplied configs.
- Checkpoints preserve model, optimizer, epoch, config, and supported RNG state.
- Exact GPU reproducibility across hardware and CUDA stacks is not claimed.
- The normal CIFAR-10 trial budget is provisional and not a final experiment.

## License

MIT. See `LICENSE`.
