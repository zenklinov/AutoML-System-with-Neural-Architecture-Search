"""Training-partition data isolation, stratification, and transforms."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

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


class TransformSubset(Dataset):
    """Apply a split-specific transform to one underlying dataset."""

    def __init__(self, dataset: Dataset, indices: list[int], transform: Any) -> None:
        self.dataset, self.indices, self.transform = dataset, indices, transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        image, target = self.dataset[self.indices[index]]
        return self.transform(image), target


def create_split_indices(
    dataset_size: int, validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Compatibility helper for synthetic data only."""
    if dataset_size < 2 or not 0 < validation_fraction < 1:
        raise ValueError("invalid split")
    size = max(1, int(dataset_size * validation_fraction))
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=g).tolist()
    return indices[size:], indices[:size]


def create_stratified_split_indices(
    labels: list[int] | torch.Tensor, validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Create deterministic, disjoint per-class splits, then shuffle each split."""
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or len(values) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("invalid stratified split")
    generator = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []
    for class_id in sorted(np.unique(values).tolist()):
        class_indices = np.flatnonzero(values == class_id)
        validation_size = int(len(class_indices) * validation_fraction)
        if validation_size < 1 or validation_size >= len(class_indices):
            raise ValueError("every class must occur in both splits")
        shuffled = generator.permutation(class_indices)
        validation_indices.extend(shuffled[:validation_size].tolist())
        train_indices.extend(shuffled[validation_size:].tolist())
    generator.shuffle(train_indices)
    generator.shuffle(validation_indices)
    train = train_indices
    validation = validation_indices
    if set(train) & set(validation) or len(train) + len(validation) != len(values):
        raise RuntimeError("split isolation invariant failed")
    return train, validation


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def prepare_search_dataset(config: DataConfig) -> None:
    if config.dataset == "cifar10":
        assert config.directory is not None
        datasets.CIFAR10(root=str(config.directory), train=True, download=True)


def _synthetic(config: DataConfig, model: ModelConfig) -> TensorDataset:
    g = torch.Generator().manual_seed(config.split_seed)
    assert config.synthetic_samples is not None
    return TensorDataset(
        torch.randn(config.synthetic_samples, model.input_channels, 32, 32, generator=g),
        torch.randint(0, model.num_classes, (config.synthetic_samples,), generator=g),
    )


def build_search_dataset(config: DataConfig, model: ModelConfig) -> Dataset:
    """Load only the official training partition, without transforms."""
    if config.dataset == "synthetic":
        return _synthetic(config, model)
    assert config.directory is not None
    return datasets.CIFAR10(root=str(config.directory), train=True, download=False, transform=None)


def build_transforms(config: DataConfig) -> tuple[Any, Any]:
    normalization = transforms.Normalize(config.normalization.mean, config.normalization.std)
    evaluation = transforms.Compose([transforms.ToTensor(), normalization])
    training = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=config.augmentation.random_crop_padding),
            transforms.RandomHorizontalFlip(config.augmentation.horizontal_flip_probability),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return training, evaluation


def get_search_data_loaders(
    data_config: DataConfig, model_config: ModelConfig, batch_size: int, training_seed: int
) -> SearchDataLoaders:
    dataset = build_search_dataset(data_config, model_config)
    if data_config.dataset == "cifar10":
        train_indices, validation_indices = create_stratified_split_indices(
            dataset.targets, data_config.validation_fraction, data_config.split_seed
        )  # type: ignore[attr-defined]
        train_transform, validation_transform = build_transforms(data_config)
        train_dataset: Dataset = TransformSubset(dataset, train_indices, train_transform)
        validation_dataset: Dataset = TransformSubset(
            dataset, validation_indices, validation_transform
        )
    else:
        train_indices, validation_indices = create_split_indices(
            len(dataset), data_config.validation_fraction, data_config.split_seed
        )
        train_dataset = Subset(dataset, train_indices)
        validation_dataset = Subset(dataset, validation_indices)
    if data_config.max_train_samples is not None:
        train_dataset = Subset(
            train_dataset, range(min(data_config.max_train_samples, len(train_dataset)))
        )
    if data_config.max_validation_samples is not None:
        validation_dataset = Subset(
            validation_dataset,
            range(min(data_config.max_validation_samples, len(validation_dataset))),
        )
    if not len(train_dataset) or not len(validation_dataset):
        raise ValueError("split must contain train and validation samples")
    generator = torch.Generator().manual_seed(training_seed)
    common = {
        "batch_size": batch_size,
        "num_workers": data_config.num_workers,
        "worker_init_fn": seed_worker if data_config.num_workers else None,
    }
    return SearchDataLoaders(
        DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        DataLoader(validation_dataset, shuffle=False, **common),
        generator,
    )


def get_full_training_loader(
    data_config: DataConfig, model_config: ModelConfig, batch_size: int, training_seed: int
) -> tuple[DataLoader, torch.Generator]:
    """Load all 50,000 official training examples; never loads the test partition."""
    dataset = build_search_dataset(data_config, model_config)
    if data_config.dataset == "cifar10":
        dataset = TransformSubset(
            dataset, list(range(len(dataset))), build_transforms(data_config)[0]
        )
    generator = torch.Generator().manual_seed(training_seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=data_config.num_workers,
        worker_init_fn=seed_worker if data_config.num_workers else None,
    )
    return loader, generator
