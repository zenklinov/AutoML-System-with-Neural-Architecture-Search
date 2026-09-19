"""Validated YAML configuration contract for NAS runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be a mapping")
    return value


def _strict_keys(mapping: dict[str, Any], allowed: set[str], field: str) -> None:
    unknown = set(mapping) - allowed
    missing = allowed - set(mapping)
    if unknown:
        raise ConfigError(f"{field} has unsupported keys: {sorted(unknown)}")
    if missing:
        raise ConfigError(f"{field} is missing keys: {sorted(missing)}")


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _optional_integer(value: Any, field: str, minimum: int = 1) -> int | None:
    if value is None:
        return None
    return _integer(value, field, minimum)


def _nonempty_choices(value: Any, field: str, allowed: set[Any]) -> tuple[Any, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field} must be a non-empty list")
    if any(choice not in allowed for choice in value):
        raise ConfigError(f"{field} contains unsupported choices; allowed: {sorted(allowed)}")
    if len(set(value)) != len(value):
        raise ConfigError(f"{field} must not contain duplicates")
    return tuple(value)


def _boolean_choices(value: Any, field: str) -> tuple[bool, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field} must be a non-empty list")
    if any(type(choice) is not bool for choice in value):
        raise ConfigError(f"{field} contains unsupported choices; allowed: [False, True]")
    if len(set(value)) != len(value):
        raise ConfigError(f"{field} must not contain duplicates")
    return tuple(value)


def _resolved_path(value: Any, field: str, base_directory: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path string")
    path = Path(value)
    return (base_directory / path).resolve() if not path.is_absolute() else path.resolve()


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    directory: Path | None
    validation_fraction: float
    split_seed: int
    num_workers: int
    max_train_samples: int | None
    max_validation_samples: int | None
    synthetic_samples: int | None


@dataclass(frozen=True)
class ArchitectureSpaceConfig:
    min_blocks: int
    max_blocks: int
    kernel_sizes: tuple[int, ...]
    output_channels: tuple[int, ...]
    activations: tuple[str, ...]
    residual: tuple[bool, ...]


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int
    num_classes: int
    search_space: ArchitectureSpaceConfig


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    max_epochs: int
    deterministic_algorithms: bool


@dataclass(frozen=True)
class SearchConfig:
    strategy: str
    seed: int
    num_trials: int
    metric: str
    mode: str
    grace_period_epochs: int
    reduction_factor: int
    max_concurrent_trials: int


@dataclass(frozen=True)
class ResourceConfig:
    cpu_per_trial: int
    gpu_per_trial: int


@dataclass(frozen=True)
class OutputConfig:
    root_directory: Path
    run_label: str


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    search: SearchConfig
    resources: ResourceConfig
    output: OutputConfig
    source_path: Path

    def to_dict(self) -> dict[str, Any]:
        """Return the resolved executable configuration as JSON-compatible data."""
        payload = asdict(self)
        payload.pop("source_path")
        payload["data"]["directory"] = (
            str(self.data.directory) if self.data.directory is not None else None
        )
        payload["output"]["root_directory"] = str(self.output.root_directory)
        for key in ("kernel_sizes", "output_channels", "activations", "residual"):
            payload["model"]["search_space"][key] = list(
                payload["model"]["search_space"][key]
            )
        return payload


def _parse_data(raw: Any, base_directory: Path) -> DataConfig:
    data = _mapping(raw, "data")
    allowed = {
        "dataset",
        "directory",
        "validation_fraction",
        "split_seed",
        "num_workers",
        "max_train_samples",
        "max_validation_samples",
        "synthetic_samples",
    }
    _strict_keys(data, allowed, "data")
    dataset = data["dataset"]
    if dataset not in {"cifar10", "synthetic"}:
        raise ConfigError("data.dataset must be 'cifar10' or 'synthetic'")
    fraction = data["validation_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, int | float):
        raise ConfigError("data.validation_fraction must be numeric")
    if not 0.0 < float(fraction) < 1.0:
        raise ConfigError("data.validation_fraction must be between 0 and 1")

    directory = data["directory"]
    if dataset == "cifar10":
        directory = _resolved_path(directory, "data.directory", base_directory)
        if data["synthetic_samples"] is not None:
            raise ConfigError("data.synthetic_samples must be null for CIFAR-10")
    else:
        if directory is not None:
            raise ConfigError("data.directory must be null for synthetic test data")
        _integer(data["synthetic_samples"], "data.synthetic_samples", 2)

    return DataConfig(
        dataset=dataset,
        directory=directory,
        validation_fraction=float(fraction),
        split_seed=_integer(data["split_seed"], "data.split_seed"),
        num_workers=_integer(data["num_workers"], "data.num_workers"),
        max_train_samples=_optional_integer(
            data["max_train_samples"], "data.max_train_samples"
        ),
        max_validation_samples=_optional_integer(
            data["max_validation_samples"], "data.max_validation_samples"
        ),
        synthetic_samples=data["synthetic_samples"],
    )


def _parse_model(raw: Any) -> ModelConfig:
    model = _mapping(raw, "model")
    _strict_keys(model, {"input_channels", "num_classes", "search_space"}, "model")
    space = _mapping(model["search_space"], "model.search_space")
    _strict_keys(
        space,
        {
            "num_blocks",
            "kernel_sizes",
            "output_channels",
            "activations",
            "residual",
        },
        "model.search_space",
    )
    depths = _mapping(space["num_blocks"], "model.search_space.num_blocks")
    _strict_keys(depths, {"min", "max"}, "model.search_space.num_blocks")
    minimum = _integer(depths["min"], "model.search_space.num_blocks.min", 2)
    maximum = _integer(depths["max"], "model.search_space.num_blocks.max", 2)
    if (minimum, maximum) != (2, 4):
        raise ConfigError("model.search_space.num_blocks must define the supported 2-4 range")

    input_channels = _integer(model["input_channels"], "model.input_channels", 1)
    num_classes = _integer(model["num_classes"], "model.num_classes", 2)
    if input_channels != 3 or num_classes != 10:
        raise ConfigError("this phase supports 3-channel, 10-class CIFAR-shaped data")
    return ModelConfig(
        input_channels=input_channels,
        num_classes=num_classes,
        search_space=ArchitectureSpaceConfig(
            min_blocks=minimum,
            max_blocks=maximum,
            kernel_sizes=_nonempty_choices(
                space["kernel_sizes"], "model.search_space.kernel_sizes", {3, 5}
            ),
            output_channels=_nonempty_choices(
                space["output_channels"],
                "model.search_space.output_channels",
                {16, 32, 64},
            ),
            activations=_nonempty_choices(
                space["activations"], "model.search_space.activations", {"relu", "silu"}
            ),
            residual=_boolean_choices(
                space["residual"], "model.search_space.residual"
            ),
        ),
    )


def _parse_training(raw: Any) -> TrainingConfig:
    training = _mapping(raw, "training")
    _strict_keys(
        training,
        {"batch_size", "learning_rate", "max_epochs", "deterministic_algorithms"},
        "training",
    )
    learning_rate = training["learning_rate"]
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, int | float):
        raise ConfigError("training.learning_rate must be numeric")
    if float(learning_rate) <= 0:
        raise ConfigError("training.learning_rate must be > 0")
    if not isinstance(training["deterministic_algorithms"], bool):
        raise ConfigError("training.deterministic_algorithms must be boolean")
    return TrainingConfig(
        batch_size=_integer(training["batch_size"], "training.batch_size", 1),
        learning_rate=float(learning_rate),
        max_epochs=_integer(training["max_epochs"], "training.max_epochs", 1),
        deterministic_algorithms=training["deterministic_algorithms"],
    )


def _parse_search(raw: Any, max_epochs: int) -> SearchConfig:
    search = _mapping(raw, "search")
    allowed = {
        "strategy",
        "seed",
        "num_trials",
        "metric",
        "mode",
        "grace_period_epochs",
        "reduction_factor",
        "max_concurrent_trials",
    }
    _strict_keys(search, allowed, "search")
    if search["strategy"] not in {"bayesian", "random"}:
        raise ConfigError("search.strategy must be 'bayesian' or 'random'")
    if search["metric"] != "validation_accuracy" or search["mode"] != "max":
        raise ConfigError("search must maximize validation_accuracy")
    grace_period = _integer(
        search["grace_period_epochs"], "search.grace_period_epochs", 1
    )
    if grace_period > max_epochs:
        raise ConfigError("search.grace_period_epochs cannot exceed training.max_epochs")
    return SearchConfig(
        strategy=search["strategy"],
        seed=_integer(search["seed"], "search.seed"),
        num_trials=_integer(search["num_trials"], "search.num_trials", 1),
        metric=search["metric"],
        mode=search["mode"],
        grace_period_epochs=grace_period,
        reduction_factor=_integer(search["reduction_factor"], "search.reduction_factor", 2),
        max_concurrent_trials=_integer(
            search["max_concurrent_trials"], "search.max_concurrent_trials", 1
        ),
    )


def _parse_resources(raw: Any) -> ResourceConfig:
    resources = _mapping(raw, "resources")
    _strict_keys(resources, {"cpu_per_trial", "gpu_per_trial"}, "resources")
    gpu = _integer(resources["gpu_per_trial"], "resources.gpu_per_trial")
    if gpu not in {0, 1}:
        raise ConfigError("resources.gpu_per_trial must be 0 or 1")
    return ResourceConfig(
        cpu_per_trial=_integer(resources["cpu_per_trial"], "resources.cpu_per_trial", 1),
        gpu_per_trial=gpu,
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML file into a validated immutable configuration."""
    source_path = Path(path).resolve()
    try:
        with source_path.open(encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file)
    except OSError as error:
        raise ConfigError(f"cannot read config {source_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"cannot parse YAML config {source_path}: {error}") from error
    root = _mapping(raw, "config")
    allowed = {"seed", "data", "model", "training", "search", "resources", "output"}
    _strict_keys(root, allowed, "config")

    training = _parse_training(root["training"])
    output = _mapping(root["output"], "output")
    _strict_keys(output, {"root_directory", "run_label"}, "output")
    run_label = output["run_label"]
    if not isinstance(run_label, str) or not run_label.strip():
        raise ConfigError("output.run_label must be a non-empty string")
    return ExperimentConfig(
        seed=_integer(root["seed"], "seed"),
        data=_parse_data(root["data"], source_path.parent),
        model=_parse_model(root["model"]),
        training=training,
        search=_parse_search(root["search"], training.max_epochs),
        resources=_parse_resources(root["resources"]),
        output=OutputConfig(
            root_directory=_resolved_path(
                output["root_directory"], "output.root_directory", source_path.parent
            ),
            run_label=run_label.strip(),
        ),
        source_path=source_path,
    )
