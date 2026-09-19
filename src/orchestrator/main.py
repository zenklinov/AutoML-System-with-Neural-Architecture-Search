"""Command-line orchestration for local multi-trial neural architecture search."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import optuna
import ray
import torch
import yaml
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from src.training.worker import prepare_cifar10, train_nas_candidate


def load_config(path: str) -> dict[str, Any]:
    """Load and minimally validate the canonical YAML experiment config."""
    with Path(path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    required_sections = {
        "environment",
        "seed",
        "data",
        "model",
        "training",
        "search",
        "resources",
        "output",
    }
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    space = config["model"]["search_space"]
    depth = space["num_blocks"]
    if int(depth["min"]) != 2 or int(depth["max"]) != 4:
        raise ValueError("Phase 1 supports a 2-4 block search space")
    if set(space["kernel_sizes"]) - {3, 5}:
        raise ValueError("Only kernel sizes 3 and 5 are implemented")
    if set(space["activations"]) - {"relu", "silu"}:
        raise ValueError("Only ReLU and SiLU are implemented")
    if config["search"]["metric"] != "validation_accuracy":
        raise ValueError("Architecture selection must use validation_accuracy")
    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply explicit run-size and device overrides without redefining the search space."""
    result = copy.deepcopy(config)
    if args.strategy is not None:
        result["search"]["strategy"] = args.strategy
    if args.trials is not None:
        result["search"]["num_trials"] = args.trials
    if args.parallelism is not None:
        result["search"]["max_concurrent_trials"] = args.parallelism
    if args.max_epochs is not None:
        result["training"]["max_epochs"] = args.max_epochs
    if args.max_train_samples is not None:
        result["data"]["max_train_samples"] = args.max_train_samples
    if args.max_validation_samples is not None:
        result["data"]["max_validation_samples"] = args.max_validation_samples
    if args.cpu:
        result["resources"]["gpu_per_trial"] = 0
    if args.gpu:
        result["resources"]["gpu_per_trial"] = 1
    return result


class ConditionalArchitectureSpace:
    """Pickle-safe Optuna define-by-run space for active CNN blocks only."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = copy.deepcopy(config)

    def __call__(self, trial: optuna.Trial) -> dict[str, Any]:
        config = self.config
        search_space = config["model"]["search_space"]
        num_blocks = trial.suggest_int(
            "num_blocks",
            int(search_space["num_blocks"]["min"]),
            int(search_space["num_blocks"]["max"]),
        )
        blocks = []
        for block_index in range(num_blocks):
            blocks.append(
                {
                    "kernel_size": trial.suggest_categorical(
                        f"block_{block_index}_kernel_size",
                        search_space["kernel_sizes"],
                    ),
                    "out_channels": trial.suggest_categorical(
                        f"block_{block_index}_out_channels",
                        search_space["output_channels"],
                    ),
                    "activation": trial.suggest_categorical(
                        f"block_{block_index}_activation",
                        search_space["activations"],
                    ),
                    "residual": trial.suggest_categorical(
                        f"block_{block_index}_residual",
                        search_space["residual"],
                    ),
                }
            )

        return {
            "architecture": {"num_blocks": num_blocks, "blocks": blocks},
            "model": {
                "input_channels": config["model"]["input_channels"],
                "num_classes": config["model"]["num_classes"],
            },
            "training": copy.deepcopy(config["training"]),
            "data": copy.deepcopy(config["data"]),
            "resources": copy.deepcopy(config["resources"]),
            "seed": config["seed"],
        }


def short_trial_name(trial: Any) -> str:
    """Keep Windows trial paths below the legacy 260-character limit."""
    return f"candidate_{trial.trial_id}"


def create_search_algorithm(config: dict[str, Any]) -> OptunaSearch:
    """Create seeded TPE or random sampling over the same conditional space."""
    seed = int(config["seed"])
    strategy = config["search"]["strategy"]
    if strategy == "bayesian":
        sampler = optuna.samplers.TPESampler(seed=seed)
    elif strategy == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError(f"Unsupported search strategy: {strategy}")
    return OptunaSearch(
        space=ConditionalArchitectureSpace(config),
        metric=config["search"]["metric"],
        mode=config["search"]["mode"],
        sampler=sampler,
    )


def run_nas(config: dict[str, Any]) -> tune.ResultGrid:
    """Run a local Ray Tune architecture search with explicit trial resources."""
    config = copy.deepcopy(config)
    # Ray changes the working directory for each trial, so shared inputs must
    # be absolute before the config is dispatched to trial processes.
    config["data"]["directory"] = str(Path(config["data"]["directory"]).resolve())
    resources = config["resources"]
    cpu_per_trial = float(resources["cpu_per_trial"])
    gpu_per_trial = float(resources["gpu_per_trial"])
    parallelism = int(config["search"]["max_concurrent_trials"])
    if gpu_per_trial > 0 and not torch.cuda.is_available():
        raise RuntimeError("GPU mode was requested, but CUDA is unavailable")

    prepare_cifar10(config["data"]["directory"])
    ray.init(
        num_cpus=max(1, int(cpu_per_trial * parallelism)),
        num_gpus=torch.cuda.device_count() if gpu_per_trial > 0 else 0,
        include_dashboard=False,
    )
    try:
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric=config["search"]["metric"],
            mode=config["search"]["mode"],
            max_t=int(config["training"]["max_epochs"]),
            grace_period=min(
                int(config["search"]["grace_period_epochs"]),
                int(config["training"]["max_epochs"]),
            ),
            reduction_factor=int(config["search"]["reduction_factor"]),
        )
        trainable = tune.with_resources(
            train_nas_candidate,
            resources={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
        )
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                search_alg=create_search_algorithm(config),
                scheduler=scheduler,
                num_samples=int(config["search"]["num_trials"]),
                max_concurrent_trials=parallelism,
                trial_name_creator=short_trial_name,
                trial_dirname_creator=short_trial_name,
            ),
            run_config=train.RunConfig(
                name=config["output"]["experiment_name"],
                storage_path=str(Path(config["output"]["storage_path"]).resolve()),
            ),
        )
        results = tuner.fit()
        best_result = results.get_best_result(
            metric=config["search"]["metric"],
            mode=config["search"]["mode"],
        )
        print("Best validation candidate:")
        print(f"  validation_accuracy: {best_result.metrics['validation_accuracy']:.4f}")
        print(f"  architecture: {best_result.config['architecture']}")
        print(f"  checkpoint: {best_result.checkpoint}")
        return results
    finally:
        ray.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local multi-trial neural architecture search")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--strategy", choices=["random", "bayesian"])
    parser.add_argument("--trials", type=int)
    parser.add_argument("--parallelism", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--cpu", action="store_true")
    device_group.add_argument("--gpu", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run_nas(apply_cli_overrides(load_config(arguments.config), arguments))
