from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml
from conftest import REPOSITORY_ROOT, make_architecture
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from automl_nas.data import build_transforms
from automl_nas.locked_test import run_locked_test_evaluation
from automl_nas.models import CandidateCNN
from automl_nas.protocol import REFERENCE_BASELINE_PARAMETER_COUNT, trainable_parameter_count
from automl_nas.search import create_search_algorithm
from automl_nas.workflows import (
    load_calibration_panel,
    run_calibration,
    run_confirmation,
    run_final_training,
)


def test_frozen_transforms_and_statistics(smoke_config) -> None:
    training, evaluation = build_transforms(smoke_config.data)
    assert [type(item) for item in training.transforms] == [
        transforms.RandomCrop,
        transforms.RandomHorizontalFlip,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert [type(item) for item in evaluation.transforms] == [
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert tuple(evaluation.transforms[-1].mean) == smoke_config.data.normalization.mean
    assert tuple(evaluation.transforms[-1].std) == smoke_config.data.normalization.std


def test_tpe_defaults_are_explicit(smoke_config) -> None:
    from dataclasses import replace

    config = replace(smoke_config, search=replace(smoke_config.search, strategy="bayesian"))
    sampler = create_search_algorithm(config)._sampler
    assert sampler._n_startup_trials == 10
    assert sampler._n_ei_candidates == 24
    assert sampler._multivariate is False
    assert sampler._constant_liar is False


def test_baseline_parameter_count_is_stable() -> None:
    from automl_nas.protocol import REFERENCE_BASELINE

    baseline = yaml.safe_load(
        (REPOSITORY_ROOT / "configs" / "fixed_reference_baseline.yaml").read_text(encoding="utf-8")
    )
    assert baseline["name"] == "fixed reference baseline"
    assert baseline["architecture"] == REFERENCE_BASELINE
    assert REFERENCE_BASELINE_PARAMETER_COUNT == 68_458
    assert trainable_parameter_count(make_architecture(2)) > 0


def test_calibration_panel_architectures_construct() -> None:
    panel = load_calibration_panel(REPOSITORY_ROOT / "configs" / "asha_calibration_panel.yaml")
    assert len(panel) == 6
    assert {item["num_blocks"] for item in panel} == {2, 3, 4}
    for architecture in panel:
        CandidateCNN(architecture)


def test_calibration_persists_completed_records_before_failure(
    monkeypatch, smoke_config, tmp_path: Path
) -> None:
    panel = tmp_path / "panel.yaml"
    panel.write_text(
        yaml.safe_dump({"architectures": [make_architecture(2), make_architecture(3)]}),
        encoding="utf-8",
    )
    calls = 0

    def train(config, architecture, seed):
        nonlocal calls
        del config, seed
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return {"architecture": architecture, "seed": 4242, "history": []}

    monkeypatch.setattr("automl_nas.workflows.prepare_search_dataset", lambda config: None)
    monkeypatch.setattr("automl_nas.workflows._train_validation", train)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_calibration(smoke_config, panel)

    summary = next(smoke_config.output.root_directory.glob("runs/*/summaries/calibration.json"))
    document = json.loads(summary.read_text(encoding="utf-8"))
    manifest = json.loads((summary.parents[1] / "manifest.json").read_text(encoding="utf-8"))
    assert document["status"] == "FAILED"
    assert len(document["records"]) == 1
    assert manifest["status"] == "FAILED"


def test_confirmation_uses_only_completed_trials_and_confirmation_seeds(
    monkeypatch, smoke_config, tmp_path: Path
) -> None:
    source = tmp_path / "trials.json"
    source.write_text(
        json.dumps(
            {
                "trials": [
                    {
                        "status": "COMPLETED",
                        "validation_accuracy": 0.5,
                        "architecture": make_architecture(2),
                    },
                    {
                        "status": "PRUNED",
                        "validation_accuracy": 0.9,
                        "architecture": make_architecture(4),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("automl_nas.workflows.prepare_search_dataset", lambda config: None)
    monkeypatch.setattr(
        "automl_nas.workflows._train_validation",
        lambda config, architecture, seed: {
            "architecture": architecture,
            "seed": seed,
            "best_validation_accuracy": seed / 10000,
            "parameter_count": 1,
            "history": [],
            "duration_seconds": 0,
        },
    )
    output = run_confirmation(smoke_config, [source], shortlist_size=3)
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["uses_asha"] is False and document["official_test_access"] is False
    assert all(
        {run["seed"] for run in item["runs"]} == set(smoke_config.protocol.confirmation_seeds)
        for item in document["ranked_candidates"]
    )
    assert all(item["architecture"]["num_blocks"] != 4 for item in document["ranked_candidates"])


def test_final_training_is_locked_and_test_requires_ack(smoke_config, tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(
        json.dumps({"stage": "final_training", "ranked_candidates": []}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="final training is disabled"):
        run_final_training(smoke_config, path)
    with pytest.raises(PermissionError, match="acknowledge"):
        run_locked_test_evaluation(smoke_config, path, False)


def test_final_training_preserves_strategy_winners_and_seeds(
    monkeypatch, smoke_config, tmp_path: Path
) -> None:
    from dataclasses import replace

    config = replace(
        smoke_config,
        protocol=replace(smoke_config.protocol, final_budget_locked=True),
    )
    confirmation = tmp_path / "confirmation.json"
    confirmation.write_text(
        json.dumps(
            {
                "strategy_winners": {
                    "random": {"architecture": make_architecture(2)},
                    "bayesian": {"architecture": make_architecture(4)},
                },
                "overall_finalist": {"architecture": make_architecture(2)},
            }
        ),
        encoding="utf-8",
    )
    tiny = TensorDataset(torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))
    monkeypatch.setattr("automl_nas.workflows.prepare_search_dataset", lambda config: None)
    monkeypatch.setattr(
        "automl_nas.workflows.get_full_training_loader",
        lambda *args: (DataLoader(tiny, batch_size=2), torch.Generator()),
    )
    monkeypatch.setattr("automl_nas.workflows.train_one_epoch", lambda *args: 1.0)
    monkeypatch.setattr(
        "automl_nas.workflows.torch.save", lambda payload, path: path.write_bytes(b"checkpoint")
    )
    document = json.loads(run_final_training(config, confirmation).read_text(encoding="utf-8"))
    assert {item["label"] for item in document["records"]} == {
        "random_selected",
        "bayesian_selected",
        "reference_baseline",
    }
    for label in {item["label"] for item in document["records"]}:
        assert {item["seed"] for item in document["records"] if item["label"] == label} == set(
            config.protocol.final_training_seeds
        )


@pytest.mark.parametrize(
    "name",
    ["scheduler_calibration.yaml", "pilot_random.yaml", "pilot_tpe.yaml"],
)
def test_provisional_protocol_templates_validate(name: str) -> None:
    from automl_nas.config import load_config

    config = load_config(REPOSITORY_ROOT / "configs" / name)
    assert config.protocol.final_budget_locked is False
    assert config.search.max_concurrent_trials == 1


def test_only_locked_module_requests_official_test_partition() -> None:
    import automl_nas.data as data_module
    import automl_nas.locked_test as locked_module
    import automl_nas.search as search_module
    import automl_nas.workflows as workflow_module

    assert "train=False" not in inspect.getsource(data_module)
    assert "train=False" not in inspect.getsource(search_module)
    assert "train=False" not in inspect.getsource(workflow_module)
    assert "train=False" in inspect.getsource(locked_module)
