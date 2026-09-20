"""Conditional search space and local Ray Tune orchestration."""

from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import ray
import torch
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from automl_nas.artifacts import (
    build_manifest,
    build_result_document,
    create_run_id,
    finalize_manifest,
    write_json,
)
from automl_nas.config import ExperimentConfig
from automl_nas.data import prepare_search_dataset
from automl_nas.training import train_nas_candidate


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    run_directory: Path
    manifest_path: Path
    result_path: Path
    best_trial_id: str


class ConditionalArchitectureSpace:
    """Pickle-safe Optuna define-by-run space that samples active blocks only."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def __call__(self, trial: optuna.Trial) -> dict[str, Any]:
        space = self.config.model.search_space
        number_of_blocks = trial.suggest_int("num_blocks", space.min_blocks, space.max_blocks)
        blocks = []
        for block_index in range(number_of_blocks):
            blocks.append(
                {
                    "kernel_size": trial.suggest_categorical(
                        f"block_{block_index}_kernel_size", space.kernel_sizes
                    ),
                    "out_channels": trial.suggest_categorical(
                        f"block_{block_index}_out_channels", space.output_channels
                    ),
                    "activation": trial.suggest_categorical(
                        f"block_{block_index}_activation", space.activations
                    ),
                    "residual": trial.suggest_categorical(
                        f"block_{block_index}_residual", space.residual
                    ),
                }
            )
        executable = self.config.to_dict()
        return {
            "architecture": {"num_blocks": number_of_blocks, "blocks": blocks},
            "model": {
                "input_channels": executable["model"]["input_channels"],
                "num_classes": executable["model"]["num_classes"],
            },
            "training": copy.deepcopy(executable["training"]),
            "data": copy.deepcopy(executable["data"]),
            "resources": copy.deepcopy(executable["resources"]),
            "seed": executable["seed"],
        }


def short_trial_name(trial: Any) -> str:
    return f"candidate_{trial.trial_id}"


def create_search_algorithm(config: ExperimentConfig) -> OptunaSearch:
    """Create a seeded TPE or random sampler over the same conditional space."""
    sampler: optuna.samplers.BaseSampler
    if config.search.strategy == "bayesian":
        sampler = optuna.samplers.TPESampler(
            seed=config.search.seed,
            n_startup_trials=config.search.tpe.n_startup_trials,
            n_ei_candidates=config.search.tpe.n_ei_candidates,
            multivariate=config.search.tpe.multivariate,
            constant_liar=config.search.tpe.constant_liar,
        )
    else:
        sampler = optuna.samplers.RandomSampler(seed=config.search.seed)
    return OptunaSearch(
        space=ConditionalArchitectureSpace(config),
        metric=config.search.metric,
        mode=config.search.mode,
        sampler=sampler,
    )


def build_scheduler(config: ExperimentConfig) -> ASHAScheduler:
    return ASHAScheduler(
        time_attr=config.search.time_attr,
        metric=config.search.metric,
        mode=config.search.mode,
        max_t=config.search.max_t,
        grace_period=config.search.grace_period_epochs,
        reduction_factor=config.search.reduction_factor,
    )


def trial_resources(config: ExperimentConfig) -> dict[str, int]:
    return {
        "cpu": config.resources.cpu_per_trial,
        "gpu": config.resources.gpu_per_trial,
    }


def run_search(
    config: ExperimentConfig,
    command: list[str] | None = None,
) -> RunOutcome:
    """Run a local search and create manifest plus canonical trial summaries."""
    run_id = create_run_id(config.output.run_label)
    run_directory = config.output.root_directory / "runs" / run_id
    manifest_path = run_directory / "manifest.json"
    result_path = run_directory / "summaries" / "trials.json"
    raw_ray_path = run_directory / "ray"
    manifest = build_manifest(config, run_id, command=command or sys.argv)
    write_json(manifest_path, manifest)
    search_start = time.perf_counter()

    ray_started_here = not ray.is_initialized()
    try:
        if config.resources.gpu_per_trial and not torch.cuda.is_available():
            raise RuntimeError("GPU mode was requested, but CUDA is unavailable")
        prepare_search_dataset(config.data)
        if ray_started_here:
            ray.init(
                num_cpus=max(
                    1,
                    config.resources.cpu_per_trial * config.search.max_concurrent_trials,
                ),
                num_gpus=(torch.cuda.device_count() if config.resources.gpu_per_trial else 0),
                include_dashboard=False,
            )
        trainable = tune.with_resources(
            train_nas_candidate,
            resources=trial_resources(config),
        )
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                search_alg=create_search_algorithm(config),
                scheduler=build_scheduler(config),
                num_samples=config.search.num_trials,
                max_concurrent_trials=config.search.max_concurrent_trials,
                trial_name_creator=short_trial_name,
                trial_dirname_creator=short_trial_name,
            ),
            run_config=train.RunConfig(
                name="tune",
                storage_path=str(raw_ray_path),
            ),
        )
        results = tuner.fit()
        result_document = build_result_document(
            results,
            run_id,
            config,
            search_duration_seconds=time.perf_counter() - search_start,
        )
        write_json(result_path, result_document)
        best_result = results.get_best_result(
            metric=config.search.metric,
            mode=config.search.mode,
        )
        best_trial_id = str(best_result.metrics.get("trial_id") or Path(best_result.path).name)
        finalize_manifest(manifest, manifest_path, "COMPLETED")
        print(f"Run ID: {run_id}")
        print(f"Manifest: {manifest_path}")
        print(f"Trial summary: {result_path}")
        print(f"Best validation trial: {best_trial_id}")
        return RunOutcome(
            run_id=run_id,
            run_directory=run_directory,
            manifest_path=manifest_path,
            result_path=result_path,
            best_trial_id=best_trial_id,
        )
    except BaseException as error:
        finalize_manifest(manifest, manifest_path, "FAILED", error)
        raise
    finally:
        if ray_started_here and ray.is_initialized():
            ray.shutdown()
