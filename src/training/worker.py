"""Ray Tune trainable and deterministic CIFAR-10 search data handling."""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any

import torch
from ray import train
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.nas.search_space import CandidateCNN


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
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    return train_indices, validation_indices


def prepare_cifar10(data_directory: str) -> None:
    """Download CIFAR-10 once before parallel trials start."""
    datasets.CIFAR10(root=data_directory, train=True, download=True)


def get_search_data_loaders(
    data_config: dict[str, Any],
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, torch.Generator]:
    """Build loaders from CIFAR-10 training data only; the test set is untouched."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(
        root=data_config["directory"],
        train=True,
        download=False,
        transform=transform,
    )
    train_indices, validation_indices = create_split_indices(
        len(dataset),
        float(data_config["validation_fraction"]),
        seed,
    )

    max_train_samples = data_config.get("max_train_samples")
    max_validation_samples = data_config.get("max_validation_samples")
    if max_train_samples is not None:
        train_indices = train_indices[: int(max_train_samples)]
    if max_validation_samples is not None:
        validation_indices = validation_indices[: int(max_validation_samples)]

    loader_generator = torch.Generator().manual_seed(seed)
    common_loader_options = {
        "batch_size": batch_size,
        "num_workers": int(data_config.get("num_workers", 0)),
    }
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        shuffle=True,
        generator=loader_generator,
        **common_loader_options,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        shuffle=False,
        **common_loader_options,
    )
    return train_loader, validation_loader, loader_generator


def save_training_state(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    completed_epoch: int,
    config: dict[str, Any],
    loader_generator: torch.Generator,
) -> None:
    """Persist state needed to continue the next epoch of a trial."""
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "completed_epoch": completed_epoch,
        "config": config,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "loader_rng_state": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def restore_training_state(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loader_generator: torch.Generator,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    """Restore a trial and return the next epoch plus its recorded config."""
    # Checkpoints are trusted artifacts produced by this code path and include
    # optimizer/RNG metadata in addition to tensors.
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)

    random.setstate(state["python_rng_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    loader_generator.set_state(state["loader_rng_state"].cpu())
    if device.type == "cuda" and "cuda_rng_state" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return int(state["completed_epoch"]) + 1, state["config"]


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_nas_candidate(config: dict[str, Any]) -> None:
    """Train and report one discrete candidate in a multi-trial NAS run."""
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    use_gpu = float(config["resources"]["gpu_per_trial"]) > 0
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    if use_gpu and device.type != "cuda":
        raise RuntimeError("This trial requested a GPU, but CUDA is unavailable")

    model = CandidateCNN(
        architecture=config["architecture"],
        input_channels=int(config["model"]["input_channels"]),
        num_classes=int(config["model"]["num_classes"]),
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
    )
    criterion = nn.CrossEntropyLoss()
    train_loader, validation_loader, loader_generator = get_search_data_loaders(
        config["data"],
        batch_size=int(config["training"]["batch_size"]),
        seed=seed,
    )

    start_epoch = 0
    checkpoint = train.get_checkpoint()
    if checkpoint is not None:
        with checkpoint.as_directory() as checkpoint_directory:
            start_epoch, checkpoint_config = restore_training_state(
                Path(checkpoint_directory) / "training_state.pt",
                model,
                optimizer,
                loader_generator,
                device,
            )
        if checkpoint_config != config:
            raise ValueError("Checkpoint configuration does not match the active trial")

    max_epochs = int(config["training"]["max_epochs"])
    for epoch in range(start_epoch, max_epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)

        validation_loss, validation_accuracy = _evaluate(model, validation_loader, device)
        metrics = {
            "epoch": epoch + 1,
            "train_loss": total_loss / total_examples,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        with tempfile.TemporaryDirectory() as checkpoint_directory:
            checkpoint_path = Path(checkpoint_directory) / "training_state.pt"
            save_training_state(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                config,
                loader_generator,
            )
            train.report(
                metrics,
                checkpoint=train.Checkpoint.from_directory(checkpoint_directory),
            )
