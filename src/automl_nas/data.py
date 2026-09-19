"""Dataset isolation and deterministic search data loaders."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

from automl_nas.config import DataConfig, ModelConfig


@dataclass(frozen=True)
class SearchDataLoaders:
    train: DataLoader
    validation: DataLoader
    generator: torch.Generator


def create_split_indices(
    dataset_size: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create deterministic, disjoint train and validation indices."""
    if dataset_size < 2:
        raise ValueError("dataset_size must be at least 2")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    validation_size = max(1, int(dataset_size * validation_fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    return indices[validation_size:], indices[:validation_size]


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def prepare_search_dataset(config: DataConfig) -> None:
    """Download only the CIFAR-10 training archive before trials start."""
    if config.dataset == "cifar10":
        assert config.directory is not None
        datasets.CIFAR10(root=str(config.directory), train=True, download=True)


def _synthetic_dataset(config: DataConfig, model: ModelConfig) -> TensorDataset:
    generator = torch.Generator().manual_seed(config.split_seed)
    assert config.synthetic_samples is not None
    images = torch.randn(
        config.synthetic_samples,
        model.input_channels,
        32,
        32,
        generator=generator,
    )
    labels = torch.randint(
        low=0,
        high=model.num_classes,
        size=(config.synthetic_samples,),
        generator=generator,
    )
    return TensorDataset(images, labels)


def build_search_dataset(config: DataConfig, model: ModelConfig) -> Dataset:
    """Return search data; CIFAR-10 always uses its official training partition."""
    if config.dataset == "synthetic":
        return _synthetic_dataset(config, model)
    assert config.directory is not None
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return datasets.CIFAR10(
        root=str(config.directory),
        train=True,
        download=False,
        transform=transform,
    )


def get_search_data_loaders(
    data_config: DataConfig,
    model_config: ModelConfig,
    batch_size: int,
    training_seed: int,
) -> SearchDataLoaders:
    """Build deterministic loaders without accessing an official test partition."""
    dataset = build_search_dataset(data_config, model_config)
    train_indices, validation_indices = create_split_indices(
        len(dataset), data_config.validation_fraction, data_config.split_seed
    )
    if data_config.max_train_samples is not None:
        train_indices = train_indices[: data_config.max_train_samples]
    if data_config.max_validation_samples is not None:
        validation_indices = validation_indices[: data_config.max_validation_samples]
    if not train_indices or not validation_indices:
        raise ValueError("configured search split must contain train and validation samples")

    generator = torch.Generator().manual_seed(training_seed)
    common_options = {
        "batch_size": batch_size,
        "num_workers": data_config.num_workers,
        "worker_init_fn": seed_worker if data_config.num_workers else None,
    }
    return SearchDataLoaders(
        train=DataLoader(
            Subset(dataset, train_indices),
            shuffle=True,
            generator=generator,
            **common_options,
        ),
        validation=DataLoader(
            Subset(dataset, validation_indices),
            shuffle=False,
            **common_options,
        ),
        generator=generator,
    )
