"""Calibration, confirmation, and final-training stages (never official test)."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from functools import cmp_to_key
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn, optim

from automl_nas.artifacts import build_manifest, create_run_id, finalize_manifest, write_json
from automl_nas.config import ExperimentConfig
from automl_nas.data import (
    get_full_training_loader,
    get_search_data_loaders,
    prepare_search_dataset,
)
from automl_nas.models import CandidateCNN
from automl_nas.protocol import REFERENCE_BASELINE, architecture_id, trainable_parameter_count
from automl_nas.training import evaluate, set_global_seed, train_one_epoch


def _device(config: ExperimentConfig) -> torch.device:
    if config.resources.gpu_per_trial and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but CUDA unavailable")
    return torch.device("cuda" if config.resources.gpu_per_trial else "cpu")


def _train_validation(
    config: ExperimentConfig, architecture: dict[str, Any], seed: int
) -> dict[str, Any]:
    set_global_seed(seed, config.training.deterministic_algorithms)
    device = _device(config)
    model = CandidateCNN(architecture, config.model.input_channels, config.model.num_classes).to(
        device
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    loaders = get_search_data_loaders(config.data, config.model, config.training.batch_size, seed)
    start = time.perf_counter()
    history = []
    for epoch in range(1, config.training.max_epochs + 1):
        train_loss = train_one_epoch(model, loaders.train, optimizer, criterion, device)
        validation_loss, validation_accuracy = evaluate(
            model, loaders.validation, criterion, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "elapsed_seconds": time.perf_counter() - start,
                "wall_clock_time_utc": datetime.now(UTC).isoformat(),
            }
        )
    return {
        "architecture": architecture,
        "architecture_id": architecture_id(architecture),
        "seed": seed,
        "parameter_count": trainable_parameter_count(architecture),
        "history": history,
        "best_validation_accuracy": max(row["validation_accuracy"] for row in history),
        "final_validation_accuracy": history[-1]["validation_accuracy"],
        "duration_seconds": time.perf_counter() - start,
    }


def _candidate_architectures(result_paths: Iterable[Path]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for path in result_paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        for trial in document["trials"]:
            if trial["status"] != "COMPLETED":
                continue
            identity = architecture_id(trial["architecture"])
            score = float(trial["validation_accuracy"])
            strategy = str(trial.get("search_strategy", "unspecified"))
            if identity not in candidates:
                candidates[identity] = {
                    "score": score,
                    "architecture": trial["architecture"],
                    "source_strategies": {strategy},
                }
            else:
                candidates[identity]["source_strategies"].add(strategy)
                candidates[identity]["score"] = max(candidates[identity]["score"], score)
    baseline_id = architecture_id(REFERENCE_BASELINE)
    if baseline_id in candidates:
        candidates[baseline_id]["source_strategies"].add("fixed_reference_baseline")
    else:
        candidates[baseline_id] = {
            "score": float("-inf"),
            "architecture": REFERENCE_BASELINE,
            "source_strategies": {"fixed_reference_baseline"},
        }
    if not candidates:
        raise ValueError("no completed search candidates were supplied")
    ranked = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
    for item in ranked:
        item["source_strategies"] = sorted(item["source_strategies"])
    return ranked


def _confirmation_compare(left: dict[str, Any], right: dict[str, Any]) -> int:
    difference = left["mean_validation_accuracy"] - right["mean_validation_accuracy"]
    if abs(difference) > 0.001:
        return -1 if difference > 0 else 1
    if left["parameter_count"] != right["parameter_count"]:
        return -1 if left["parameter_count"] < right["parameter_count"] else 1
    return -1 if left["architecture_id"] < right["architecture_id"] else 1


def run_confirmation(
    config: ExperimentConfig, result_paths: Iterable[Path], shortlist_size: int = 3
) -> Path:
    """Retrain shortlisted completed candidates without Ray, ASHA, or test access."""
    if shortlist_size < 3:
        raise ValueError("shortlist_size must preserve Random, TPE, and the reference baseline")
    run_id = create_run_id("confirmation")
    root = config.output.root_directory / "runs" / run_id
    manifest_path = root / "manifest.json"
    manifest = build_manifest(
        config,
        run_id,
        entry_point="automl-nas confirm",
        artifacts={"confirmation_summary": "summaries/confirmation.json"},
    )
    write_json(manifest_path, manifest)
    prepare_search_dataset(config.data)
    candidates = _candidate_architectures(result_paths)
    required = []
    for strategy in ("random", "bayesian", "fixed_reference_baseline"):
        candidate = next(
            (item for item in candidates if strategy in item["source_strategies"]), None
        )
        if candidate is not None and candidate not in required:
            required.append(candidate)
    candidates = (required + [item for item in candidates if item not in required])[:shortlist_size]
    records = []
    for candidate in candidates:
        architecture = candidate["architecture"]
        runs = [
            _train_validation(config, architecture, seed)
            for seed in config.protocol.confirmation_seeds
        ]
        scores = [run["best_validation_accuracy"] for run in runs]
        records.append(
            {
                "architecture": architecture,
                "architecture_id": architecture_id(architecture),
                "parameter_count": trainable_parameter_count(architecture),
                "source_strategies": candidate["source_strategies"],
                "runs": runs,
                "mean_validation_accuracy": statistics.fmean(scores),
                "std_validation_accuracy": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            }
        )
    records.sort(key=cmp_to_key(_confirmation_compare))
    strategy_winners = {}
    for strategy in ("random", "bayesian"):
        winner = next((row for row in records if strategy in row["source_strategies"]), None)
        if winner is not None:
            strategy_winners[strategy] = {
                "architecture": winner["architecture"],
                "architecture_id": winner["architecture_id"],
                "mean_validation_accuracy": winner["mean_validation_accuracy"],
                "parameter_count": winner["parameter_count"],
            }
    output = root / "summaries" / "confirmation.json"
    write_json(
        output,
        {
            "schema_version": 1,
            "stage": "confirmation",
            "run_id": run_id,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "source_results": [str(p) for p in result_paths],
            "seeds": list(config.protocol.confirmation_seeds),
            "uses_asha": False,
            "official_test_access": False,
            "strategy_winners": strategy_winners,
            "overall_finalist": next(
                row for row in records if "fixed_reference_baseline" not in row["source_strategies"]
            ),
            "ranked_candidates": records,
        },
    )
    finalize_manifest(manifest, manifest_path, "COMPLETED")
    return output


def load_calibration_panel(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        set(raw) != {"architectures"}
        or not isinstance(raw["architectures"], list)
        or not raw["architectures"]
    ):
        raise ValueError("calibration panel requires architectures")
    return raw["architectures"]


def run_calibration(config: ExperimentConfig, panel_path: Path) -> Path:
    """Collect full learning curves for a fixed panel; ASHA is intentionally absent."""
    run_id = create_run_id("asha-calibration")
    root = config.output.root_directory / "runs" / run_id
    manifest_path = root / "manifest.json"
    manifest = build_manifest(
        config,
        run_id,
        entry_point="automl-nas calibrate-asha",
        artifacts={"calibration_summary": "summaries/calibration.json"},
    )
    write_json(manifest_path, manifest)
    prepare_search_dataset(config.data)
    records = [
        _train_validation(config, a, config.protocol.search_training_seed)
        for a in load_calibration_panel(panel_path)
    ]
    output = root / "summaries" / "calibration.json"
    write_json(
        output,
        {
            "schema_version": 1,
            "stage": "asha_calibration",
            "run_id": run_id,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "panel_source": str(panel_path),
            "uses_asha": False,
            "official_test_access": False,
            "records": records,
        },
    )
    finalize_manifest(manifest, manifest_path, "COMPLETED")
    return output


def run_final_training(config: ExperimentConfig, confirmation_path: Path) -> Path:
    """Retrain the selected candidate and baseline on all official training data."""
    if not config.protocol.final_budget_locked:
        raise RuntimeError(
            "final training is disabled until protocol.final_budget_locked is explicitly set true"
        )
    document = json.loads(confirmation_path.read_text(encoding="utf-8"))
    selected = [
        (f"{strategy}_selected", item["architecture"])
        for strategy, item in document["strategy_winners"].items()
    ]
    if not selected:
        selected = [("overall_nas_finalist", document["overall_finalist"]["architecture"])]
    prepare_search_dataset(config.data)
    run_id = create_run_id("final-training")
    root = config.output.root_directory / "runs" / run_id
    manifest_path = root / "manifest.json"
    manifest = build_manifest(
        config,
        run_id,
        entry_point="automl-nas final-train",
        artifacts={
            "final_models": "summaries/final_models.json",
            "checkpoints": "checkpoints/",
        },
    )
    write_json(manifest_path, manifest)
    records = []
    for label, architecture in (*selected, ("reference_baseline", REFERENCE_BASELINE)):
        for seed in config.protocol.final_training_seeds:
            set_global_seed(seed, config.training.deterministic_algorithms)
            device = _device(config)
            model = CandidateCNN(architecture, 3, 10).to(device)
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
            criterion = nn.CrossEntropyLoss()
            loader, generator = get_full_training_loader(
                config.data, config.model, config.training.batch_size, seed
            )
            start = time.perf_counter()
            history = []
            for epoch in range(1, config.training.max_epochs + 1):
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_one_epoch(model, loader, optimizer, criterion, device),
                        "elapsed_seconds": time.perf_counter() - start,
                        "wall_clock_time_utc": datetime.now(UTC).isoformat(),
                    }
                )
            checkpoint = root / "checkpoints" / f"{label}-{seed}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": architecture,
                    "seed": seed,
                    "completed_epochs": config.training.max_epochs,
                    "loader_rng_state": generator.get_state(),
                    "config": config.to_dict(),
                },
                checkpoint,
            )
            records.append(
                {
                    "label": label,
                    "architecture": architecture,
                    "architecture_id": architecture_id(architecture),
                    "parameter_count": trainable_parameter_count(architecture),
                    "seed": seed,
                    "history": history,
                    "duration_seconds": time.perf_counter() - start,
                    "checkpoint": str(checkpoint),
                }
            )
    output = root / "summaries" / "final_models.json"
    write_json(
        output,
        {
            "schema_version": 1,
            "stage": "final_training",
            "run_id": run_id,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "official_training_examples": 50000
            if config.data.dataset == "cifar10"
            else config.data.synthetic_samples,
            "official_test_access": False,
            "records": records,
        },
    )
    finalize_manifest(manifest, manifest_path, "COMPLETED")
    return output
