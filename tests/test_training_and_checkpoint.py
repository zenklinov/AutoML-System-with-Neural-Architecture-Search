from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
from conftest import make_architecture
from torch import nn

from automl_nas.data import get_search_data_loaders
from automl_nas.models import CandidateCNN
from automl_nas.training import (
    restore_training_state,
    save_training_state,
    set_global_seed,
    train_nas_candidate,
    train_one_epoch,
)


def test_deterministic_seed_configures_cublas_workspace(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    set_global_seed(123, deterministic_algorithms=True)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def _trial_config(config) -> dict:
    payload = config.to_dict()
    return {
        "architecture": make_architecture(2),
        "model": {
            "input_channels": config.model.input_channels,
            "num_classes": config.model.num_classes,
        },
        "training": payload["training"],
        "data": payload["data"],
        "resources": payload["resources"],
        "seed": config.seed,
    }


def test_one_tiny_training_step_succeeds(smoke_config) -> None:
    loaders = get_search_data_loaders(
        smoke_config.data,
        smoke_config.model,
        smoke_config.training.batch_size,
        smoke_config.seed,
    )
    model = CandidateCNN(make_architecture(2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = train_one_epoch(
        model, loaders.train, optimizer, nn.CrossEntropyLoss(), torch.device("cpu")
    )
    assert loss > 0


def test_ray_trainable_reports_expected_metrics(monkeypatch, smoke_config) -> None:
    reports: list[dict] = []

    def fake_report(metrics, checkpoint) -> None:
        assert (Path(checkpoint.path) / "training_state.pt").exists()
        reports.append(metrics)

    monkeypatch.setattr("automl_nas.training.train.get_checkpoint", lambda: None)
    monkeypatch.setattr("automl_nas.training.train.report", fake_report)
    train_nas_candidate(_trial_config(smoke_config))
    assert len(reports) == 1
    assert {
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_accuracy",
    } <= set(reports[0])
    assert reports[0]["parameter_count"] > 0
    assert reports[0]["elapsed_trial_seconds"] >= 0


def test_checkpoint_round_trip_and_resume_training(smoke_config) -> None:
    architecture = make_architecture(2)
    model = CandidateCNN(architecture)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loaders = get_search_data_loaders(
        smoke_config.data,
        smoke_config.model,
        smoke_config.training.batch_size,
        smoke_config.seed,
    )
    criterion = nn.CrossEntropyLoss()
    train_one_epoch(model, loaders.train, optimizer, criterion, torch.device("cpu"))
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    config = _trial_config(smoke_config)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "training_state.pt"
        save_training_state(path, model, optimizer, 0, config, loaders.generator)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        next_epoch, restored_config = restore_training_state(
            path, model, optimizer, loaders.generator, torch.device("cpu")
        )
        assert next_epoch == 1
        assert restored_config == config
        assert optimizer.state
        for name, value in model.state_dict().items():
            assert torch.equal(value, expected[name])
        resumed_loss = train_one_epoch(
            model, loaders.train, optimizer, criterion, torch.device("cpu")
        )
        assert resumed_loss > 0
