from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from conftest import make_architecture
from torch.utils.data import TensorDataset

from automl_nas.data import (
    build_search_dataset,
    create_split_indices,
    create_stratified_split_indices,
)
from automl_nas.models import CandidateCNN, ConfigurableConvBlock
from automl_nas.training import set_global_seed


@pytest.mark.parametrize("depth", [2, 4])
def test_candidate_min_max_forward(depth: int) -> None:
    model = CandidateCNN(make_architecture(depth))
    assert model(torch.randn(3, 3, 32, 32)).shape == (3, 10)


def test_residual_projection_changes_channels() -> None:
    block = ConfigurableConvBlock(16, 32, 3, "relu", True)
    assert block.project is not None
    assert block(torch.randn(2, 16, 8, 8)).shape == (2, 32, 8, 8)


def test_split_is_deterministic_and_disjoint() -> None:
    first_train, first_validation = create_split_indices(50_000, 0.1, 2026)
    second_train, second_validation = create_split_indices(50_000, 0.1, 2026)
    assert first_train == second_train
    assert first_validation == second_validation
    assert set(first_train).isdisjoint(first_validation)
    assert (len(first_train), len(first_validation)) == (45_000, 5_000)


def test_search_never_requests_official_cifar_test_data(monkeypatch, smoke_config) -> None:
    calls: list[bool] = []

    def fake_cifar(*, root, train, download, transform=None):
        del root, download, transform
        calls.append(train)
        return TensorDataset(torch.randn(20, 3, 32, 32), torch.zeros(20, dtype=torch.long))

    cifar_config = replace(
        smoke_config.data,
        dataset="cifar10",
        directory=smoke_config.output.root_directory / "data",
        validation_fraction=0.2,
        split_seed=1,
        max_train_samples=None,
        max_validation_samples=None,
        synthetic_samples=None,
    )
    monkeypatch.setattr("automl_nas.data.datasets.CIFAR10", fake_cifar)
    build_search_dataset(cifar_config, smoke_config.model)
    assert calls == [True]


def test_stratified_split_has_exact_class_balance() -> None:
    labels = np.repeat(np.arange(10), 5_000)
    train, validation = create_stratified_split_indices(labels, 0.1, 2026)
    assert np.bincount(labels[train]).tolist() == [4_500] * 10
    assert np.bincount(labels[validation]).tolist() == [500] * 10
    assert set(train).isdisjoint(validation)


def test_global_seed_controls_python_numpy_and_torch() -> None:
    set_global_seed(42, deterministic_algorithms=True)
    first = (random.random(), np.random.rand(), torch.rand(2))
    set_global_seed(42, deterministic_algorithms=True)
    second = (random.random(), np.random.rand(), torch.rand(2))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
