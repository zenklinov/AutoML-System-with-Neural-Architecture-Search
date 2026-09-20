"""Strict immutable experiment-protocol configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def _map(v: Any, f: str) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise ConfigError(f"{f} must be a mapping")
    return v


def _keys(v: dict[str, Any], keys: set[str], f: str) -> None:
    if set(v) != keys:
        raise ConfigError(
            f"{f} missing keys={sorted(keys - set(v))}; unsupported keys={sorted(set(v) - keys)}"
        )


def _int(v: Any, f: str, minimum: int = 0) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
        raise ConfigError(f"{f} must be an integer >= {minimum}")
    return v


def _num(v: Any, f: str, minimum: float = 0.0) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or v < minimum:
        raise ConfigError(f"{f} must be numeric and >= {minimum}")
    return float(v)


def _path(v: Any, f: str, base: Path) -> Path:
    if not isinstance(v, str) or not v.strip():
        raise ConfigError(f"{f} must be a non-empty path")
    p = Path(v)
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


def _choices(v: Any, f: str, allowed: set[Any]) -> tuple[Any, ...]:
    if (
        not isinstance(v, list)
        or not v
        or any(x not in allowed for x in v)
        or len(set(v)) != len(v)
    ):
        raise ConfigError(f"{f} must be a unique non-empty subset of {sorted(allowed, key=str)}")
    return tuple(v)


def _seeds(v: Any, f: str) -> tuple[int, ...]:
    if not isinstance(v, list) or not v:
        raise ConfigError(f"{f} must be a non-empty list")
    result = tuple(_int(x, f) for x in v)
    if len(set(result)) != len(result):
        raise ConfigError(f"{f} must be unique")
    return result


@dataclass(frozen=True)
class NormalizationConfig:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    provenance: str


@dataclass(frozen=True)
class AugmentationConfig:
    random_crop_padding: int
    horizontal_flip_probability: float


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    directory: Path | None
    validation_fraction: float
    split_seed: int
    stratified: bool
    num_workers: int
    max_train_samples: int | None
    max_validation_samples: int | None
    synthetic_samples: int | None
    normalization: NormalizationConfig
    augmentation: AugmentationConfig


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
    optimizer: str
    learning_rate: float
    weight_decay: float
    lr_scheduler: str
    evaluation_interval_epochs: int
    max_epochs: int
    deterministic_algorithms: bool


@dataclass(frozen=True)
class TPEConfig:
    n_startup_trials: int
    n_ei_candidates: int
    multivariate: bool
    constant_liar: bool


@dataclass(frozen=True)
class SearchConfig:
    strategy: str
    seed: int
    num_trials: int
    metric: str
    mode: str
    time_attr: str
    max_t: int
    grace_period_epochs: int
    reduction_factor: int
    max_concurrent_trials: int
    tpe: TPEConfig


@dataclass(frozen=True)
class ResourceConfig:
    cpu_per_trial: int
    gpu_per_trial: int


@dataclass(frozen=True)
class ProtocolConfig:
    candidate_search_seeds: tuple[int, ...]
    search_training_seed: int
    confirmation_seeds: tuple[int, ...]
    final_training_seeds: tuple[int, ...]
    final_budget_locked: bool


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
    protocol: ProtocolConfig
    output: OutputConfig
    source_path: Path

    def to_dict(self) -> dict[str, Any]:
        p = asdict(self)
        p.pop("source_path")
        p["data"]["directory"] = str(self.data.directory) if self.data.directory else None
        p["output"]["root_directory"] = str(self.output.root_directory)
        for k in ("kernel_sizes", "output_channels", "activations", "residual"):
            p["model"]["search_space"][k] = list(p["model"]["search_space"][k])
        for k in ("mean", "std"):
            p["data"]["normalization"][k] = list(p["data"]["normalization"][k])
        for k in ("candidate_search_seeds", "confirmation_seeds", "final_training_seeds"):
            p["protocol"][k] = list(p["protocol"][k])
        return p


def _data(raw: Any, base: Path) -> DataConfig:
    d = _map(raw, "data")
    _keys(
        d,
        {
            "dataset",
            "directory",
            "validation_fraction",
            "split_seed",
            "stratified",
            "num_workers",
            "max_train_samples",
            "max_validation_samples",
            "synthetic_samples",
            "normalization",
            "augmentation",
        },
        "data",
    )
    if d["dataset"] not in {"cifar10", "synthetic"}:
        raise ConfigError("data.dataset must be cifar10 or synthetic")
    fraction = _num(d["validation_fraction"], "validation_fraction")
    if not 0 < fraction < 1:
        raise ConfigError("validation_fraction must be between 0 and 1")
    if not isinstance(d["stratified"], bool):
        raise ConfigError("stratified must be boolean")
    if d["dataset"] == "cifar10" and not d["stratified"]:
        raise ConfigError("CIFAR-10 split must be stratified")
    directory = d["directory"]
    if d["dataset"] == "cifar10":
        directory = _path(directory, "data.directory", base)
        if d["synthetic_samples"] is not None:
            raise ConfigError("synthetic_samples must be null for CIFAR-10")
    else:
        if directory is not None:
            raise ConfigError("directory must be null for synthetic data")
        _int(d["synthetic_samples"], "synthetic_samples", 2)
    n = _map(d["normalization"], "normalization")
    _keys(n, {"mean", "std", "provenance"}, "normalization")
    if any(not isinstance(n[k], list) or len(n[k]) != 3 for k in ("mean", "std")):
        raise ConfigError("normalization mean/std require three values")
    mean = tuple(_num(x, "mean") for x in n["mean"])
    std = tuple(_num(x, "std") for x in n["std"])
    if (
        any(x <= 0 for x in std)
        or not isinstance(n["provenance"], str)
        or not n["provenance"].strip()
    ):
        raise ConfigError("invalid normalization")
    a = _map(d["augmentation"], "augmentation")
    _keys(a, {"random_crop_padding", "horizontal_flip_probability"}, "augmentation")
    flip = _num(a["horizontal_flip_probability"], "horizontal_flip_probability")
    if flip > 1:
        raise ConfigError("horizontal_flip_probability must be <= 1")

    def optional(value: Any, field: str) -> int | None:
        return None if value is None else _int(value, field, 1)

    return DataConfig(
        d["dataset"],
        directory,
        fraction,
        _int(d["split_seed"], "split_seed"),
        d["stratified"],
        _int(d["num_workers"], "num_workers"),
        optional(d["max_train_samples"], "max_train_samples"),
        optional(d["max_validation_samples"], "max_validation_samples"),
        d["synthetic_samples"],
        NormalizationConfig(mean, std, n["provenance"]),
        AugmentationConfig(_int(a["random_crop_padding"], "random_crop_padding"), flip),
    )


def _model(raw: Any) -> ModelConfig:
    m = _map(raw, "model")
    _keys(m, {"input_channels", "num_classes", "search_space"}, "model")
    s = _map(m["search_space"], "search_space")
    _keys(
        s,
        {"num_blocks", "kernel_sizes", "output_channels", "activations", "residual"},
        "search_space",
    )
    depth = _map(s["num_blocks"], "num_blocks")
    _keys(depth, {"min", "max"}, "num_blocks")
    if (depth["min"], depth["max"]) != (2, 4):
        raise ConfigError("supported depth is 2-4")
    if (m["input_channels"], m["num_classes"]) != (3, 10):
        raise ConfigError("protocol requires 3 input channels and 10 classes")
    return ModelConfig(
        3,
        10,
        ArchitectureSpaceConfig(
            2,
            4,
            _choices(s["kernel_sizes"], "kernel_sizes", {3, 5}),
            _choices(s["output_channels"], "output_channels", {16, 32, 64}),
            _choices(s["activations"], "activations", {"relu", "silu"}),
            _choices(s["residual"], "residual", {True, False}),
        ),
    )


def _training(raw: Any) -> TrainingConfig:
    d = _map(raw, "training")
    _keys(
        d,
        {
            "batch_size",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "lr_scheduler",
            "evaluation_interval_epochs",
            "max_epochs",
            "deterministic_algorithms",
        },
        "training",
    )
    if (
        d["optimizer"] != "adam"
        or d["lr_scheduler"] != "none"
        or d["evaluation_interval_epochs"] != 1
    ):
        raise ConfigError("approved recipe requires Adam, no scheduler, and evaluation every epoch")
    if not isinstance(d["deterministic_algorithms"], bool):
        raise ConfigError("deterministic_algorithms must be boolean")
    return TrainingConfig(
        _int(d["batch_size"], "batch_size", 1),
        d["optimizer"],
        _num(d["learning_rate"], "learning_rate", 1e-15),
        _num(d["weight_decay"], "weight_decay"),
        d["lr_scheduler"],
        1,
        _int(d["max_epochs"], "max_epochs", 1),
        d["deterministic_algorithms"],
    )


def _search(raw: Any, max_epochs: int) -> SearchConfig:
    d = _map(raw, "search")
    _keys(
        d,
        {
            "strategy",
            "seed",
            "num_trials",
            "metric",
            "mode",
            "time_attr",
            "max_t",
            "grace_period_epochs",
            "reduction_factor",
            "max_concurrent_trials",
            "tpe",
        },
        "search",
    )
    if d["strategy"] not in {"bayesian", "random"}:
        raise ConfigError("strategy must be bayesian or random")
    if (d["metric"], d["mode"], d["time_attr"]) != (
        "validation_accuracy",
        "max",
        "training_iteration",
    ):
        raise ConfigError("search objective/time_attr do not match protocol")
    if d["max_t"] != max_epochs:
        raise ConfigError("search.max_t must equal training.max_epochs")
    grace = _int(d["grace_period_epochs"], "grace_period_epochs", 1)
    if grace > max_epochs:
        raise ConfigError("grace period cannot exceed max_t")
    t = _map(d["tpe"], "tpe")
    _keys(t, {"n_startup_trials", "n_ei_candidates", "multivariate", "constant_liar"}, "tpe")
    if not isinstance(t["multivariate"], bool) or not isinstance(t["constant_liar"], bool):
        raise ConfigError("TPE flags must be boolean")
    return SearchConfig(
        d["strategy"],
        _int(d["seed"], "search.seed"),
        _int(d["num_trials"], "num_trials", 1),
        d["metric"],
        d["mode"],
        d["time_attr"],
        max_epochs,
        grace,
        _int(d["reduction_factor"], "reduction_factor", 2),
        _int(d["max_concurrent_trials"], "max_concurrent_trials", 1),
        TPEConfig(
            _int(t["n_startup_trials"], "n_startup_trials", 1),
            _int(t["n_ei_candidates"], "n_ei_candidates", 1),
            t["multivariate"],
            t["constant_liar"],
        ),
    )


def _protocol(raw: Any, seed: int) -> ProtocolConfig:
    d = _map(raw, "protocol")
    _keys(
        d,
        {
            "candidate_search_seeds",
            "search_training_seed",
            "confirmation_seeds",
            "final_training_seeds",
            "final_budget_locked",
        },
        "protocol",
    )
    if not isinstance(d["final_budget_locked"], bool):
        raise ConfigError("final_budget_locked must be boolean")
    p = ProtocolConfig(
        _seeds(d["candidate_search_seeds"], "candidate_search_seeds"),
        _int(d["search_training_seed"], "search_training_seed"),
        _seeds(d["confirmation_seeds"], "confirmation_seeds"),
        _seeds(d["final_training_seeds"], "final_training_seeds"),
        d["final_budget_locked"],
    )
    if seed != p.search_training_seed:
        raise ConfigError("root seed must equal search_training_seed")
    return p


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"cannot read config {source}: {e}") from e
    try:
        root = _map(yaml.safe_load(text), "config")
    except yaml.YAMLError as e:
        raise ConfigError(f"cannot parse YAML config {source}: {e}") from e
    _keys(
        root,
        {"seed", "data", "model", "training", "search", "resources", "protocol", "output"},
        "config",
    )
    seed = _int(root["seed"], "seed")
    training = _training(root["training"])
    r = _map(root["resources"], "resources")
    _keys(r, {"cpu_per_trial", "gpu_per_trial"}, "resources")
    gpu = _int(r["gpu_per_trial"], "gpu_per_trial")
    if gpu not in {0, 1}:
        raise ConfigError("gpu_per_trial must be 0 or 1")
    o = _map(root["output"], "output")
    _keys(o, {"root_directory", "run_label"}, "output")
    if not isinstance(o["run_label"], str) or not o["run_label"].strip():
        raise ConfigError("run_label must be non-empty")
    return ExperimentConfig(
        seed,
        _data(root["data"], source.parent),
        _model(root["model"]),
        training,
        _search(root["search"], training.max_epochs),
        ResourceConfig(_int(r["cpu_per_trial"], "cpu_per_trial", 1), gpu),
        _protocol(root["protocol"], seed),
        OutputConfig(
            _path(o["root_directory"], "root_directory", source.parent), o["run_label"].strip()
        ),
        source,
    )
