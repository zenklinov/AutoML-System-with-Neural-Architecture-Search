"""Training, evaluation, checkpoint, and Ray reporting logic."""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ray import train
from torch import nn, optim
from torch.utils.data import DataLoader

from automl_nas.config import ArchitectureSpaceConfig, DataConfig, ModelConfig
from automl_nas.data import get_search_data_loaders
from automl_nas.models import CandidateCNN


def set_global_seed(seed: int, deterministic_algorithms: bool) -> None:
    """Seed supported RNGs and request deterministic kernels where practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic_algorithms


def _data_config(raw: dict[str, Any]) -> DataConfig:
    return DataConfig(
        dataset=str(raw["dataset"]),
        directory=Path(raw["directory"]) if raw["directory"] is not None else None,
        validation_fraction=float(raw["validation_fraction"]),
        split_seed=int(raw["split_seed"]),
        num_workers=int(raw["num_workers"]),
        max_train_samples=raw["max_train_samples"],
        max_validation_samples=raw["max_validation_samples"],
        synthetic_samples=raw["synthetic_samples"],
    )


def _model_config(raw: dict[str, Any]) -> ModelConfig:
    # Trial construction needs input/output dimensions; the architecture itself
    # has already been validated and sampled by the search-space boundary.
    search_space = ArchitectureSpaceConfig(
        min_blocks=2,
        max_blocks=4,
        kernel_sizes=(3, 5),
        output_channels=(16, 32, 64),
        activations=("relu", "silu"),
        residual=(True, False),
    )
    return ModelConfig(
        input_channels=int(raw["input_channels"]),
        num_classes=int(raw["num_classes"]),
        search_space=search_space,
    )


def save_training_state(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    completed_epoch: int,
    config: dict[str, Any],
    loader_generator: torch.Generator,
) -> None:
    """Persist enough state to continue at the next epoch."""
    state: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "completed_epoch": completed_epoch,
        "config": config,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
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
    """Restore training state and return the next epoch and saved config."""
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    loader_generator.set_state(state["loader_rng_state"].cpu())
    if device.type == "cuda" and "cuda_rng_state" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return int(state["completed_epoch"]) + 1, state["config"]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train a candidate for one epoch and return mean example loss."""
    model.train()
    total_loss = 0.0
    total_examples = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_examples += labels.size(0)
    if total_examples == 0:
        raise ValueError("training loader is empty")
    return total_loss / total_examples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return validation loss and accuracy without modifying model state."""
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
    if total == 0:
        raise ValueError("validation loader is empty")
    return total_loss / total, correct / total


def train_nas_candidate(config: dict[str, Any]) -> None:
    """Train and report one candidate through Ray Tune's function API."""
    seed = int(config["seed"])
    deterministic = bool(config["training"]["deterministic_algorithms"])
    set_global_seed(seed, deterministic)

    gpu_requested = int(config["resources"]["gpu_per_trial"]) == 1
    device = torch.device("cuda" if gpu_requested and torch.cuda.is_available() else "cpu")
    if gpu_requested and device.type != "cuda":
        raise RuntimeError("trial requested a GPU, but CUDA is unavailable")

    model_config = _model_config(config["model"])
    model = CandidateCNN(
        architecture=config["architecture"],
        input_channels=model_config.input_channels,
        num_classes=model_config.num_classes,
    ).to(device)
    learning_rate = float(config["training"]["learning_rate"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer.param_groups[0]["lr"] != learning_rate:
        raise RuntimeError("optimizer learning rate was not applied")
    criterion = nn.CrossEntropyLoss()
    loaders = get_search_data_loaders(
        _data_config(config["data"]),
        model_config,
        batch_size=int(config["training"]["batch_size"]),
        training_seed=seed,
    )

    start_epoch = 0
    checkpoint = train.get_checkpoint()
    if checkpoint is not None:
        with checkpoint.as_directory() as checkpoint_directory:
            start_epoch, checkpoint_config = restore_training_state(
                Path(checkpoint_directory) / "training_state.pt",
                model,
                optimizer,
                loaders.generator,
                device,
            )
        if checkpoint_config != config:
            raise ValueError("checkpoint configuration does not match the active trial")

    for epoch in range(start_epoch, int(config["training"]["max_epochs"])):
        training_loss = train_one_epoch(model, loaders.train, optimizer, criterion, device)
        validation_loss, validation_accuracy = evaluate(
            model, loaders.validation, criterion, device
        )
        metrics = {
            "epoch": epoch + 1,
            "train_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        with tempfile.TemporaryDirectory() as checkpoint_directory:
            save_training_state(
                Path(checkpoint_directory) / "training_state.pt",
                model,
                optimizer,
                epoch,
                config,
                loaders.generator,
            )
            train.report(
                metrics,
                checkpoint=train.Checkpoint.from_directory(checkpoint_directory),
            )
