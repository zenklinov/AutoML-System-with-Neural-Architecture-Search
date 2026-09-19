"""Run provenance manifests and canonical trial-summary artifacts."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import optuna
import ray
import torch
import torchvision

from automl_nas.config import ExperimentConfig

MANIFEST_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def create_run_id(label: str) -> str:
    safe_label = "".join(character if character.isalnum() else "-" for character in label)
    safe_label = "-".join(part for part in safe_label.split("-") if part).lower()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_label}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _git_state(start_path: Path) -> dict[str, Any]:
    repository = next(
        (
            candidate
            for candidate in (start_path, *start_path.parents)
            if (candidate / ".git").exists()
        ),
        None,
    )
    if repository is None:
        return {"available": False, "commit_sha": None, "dirty": None}
    git = [
        "git",
        "-c",
        f"safe.directory={repository.as_posix()}",
        "-C",
        str(repository),
    ]
    try:
        subprocess.run(
            [*git, "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = subprocess.run(
            [*git, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [*git, "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"available": True, "commit_sha": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "commit_sha": None, "dirty": None}


def build_manifest(
    config: ExperimentConfig,
    run_id: str,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Build a provenance manifest without collecting personal identifiers."""
    gpu_models = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "RUNNING",
        "started_at_utc": utc_timestamp(),
        "finished_at_utc": None,
        "git": _git_state(config.source_path.parent),
        "command": command if command is not None else sys.argv,
        "entry_point": "automl-nas search",
        "config_source": str(config.source_path),
        "config": config.to_dict(),
        "seeds": {
            "training": config.seed,
            "dataset_split": config.data.split_seed,
            "search": config.search.seed,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "ray": ray.__version__,
            "optuna": optuna.__version__,
        },
        "system": {
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "cpu": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
            "gpu_models": gpu_models,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        },
        "execution": {
            "search_strategy": config.search.strategy,
            "cpu_per_trial": config.resources.cpu_per_trial,
            "gpu_per_trial": config.resources.gpu_per_trial,
            "max_concurrent_trials": config.search.max_concurrent_trials,
            "deterministic_algorithms_requested": (
                config.training.deterministic_algorithms
            ),
        },
        "artifacts": {
            "trial_summary": "summaries/trials.json",
            "raw_ray_output": "ray/",
        },
        "error": None,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary_path.replace(path)


def finalize_manifest(
    manifest: dict[str, Any],
    path: Path,
    status: str,
    error: BaseException | None = None,
) -> None:
    manifest["status"] = status
    manifest["finished_at_utc"] = utc_timestamp()
    manifest["error"] = (
        {"type": type(error).__name__, "message": str(error)} if error else None
    )
    write_json(path, manifest)


def checkpoint_reference(checkpoint: Any) -> str | None:
    if checkpoint is None:
        return None
    return str(getattr(checkpoint, "path", checkpoint))


def build_trial_summary(
    result: Any,
    run_id: str,
    config: ExperimentConfig,
) -> dict[str, Any]:
    metrics = result.metrics or {}
    iteration = int(metrics.get("training_iteration", metrics.get("epoch", 0)))
    if result.error is not None:
        status = "ERROR"
    elif iteration < config.training.max_epochs:
        status = "PRUNED"
    else:
        status = "COMPLETED"
    trial_id = str(metrics.get("trial_id") or Path(result.path).name)
    summary = {
        "run_id": run_id,
        "trial_id": trial_id,
        "search_strategy": config.search.strategy,
        "search_seed": config.search.seed,
        "training_seed": config.seed,
        "architecture": result.config["architecture"],
        "training": result.config["training"],
        "epoch": int(metrics.get("epoch", 0)),
        "training_iteration": iteration,
        "validation_accuracy": metrics.get("validation_accuracy"),
        "validation_loss": metrics.get("validation_loss"),
        "train_loss": metrics.get("train_loss"),
        "status": status,
        "duration_seconds": metrics.get("time_total_s"),
        "checkpoint_reference": checkpoint_reference(result.checkpoint),
        "pruned": status == "PRUNED",
    }
    validate_trial(summary)
    return summary


def validate_trial(trial: dict[str, Any]) -> None:
    required = {
        "run_id",
        "trial_id",
        "search_strategy",
        "search_seed",
        "training_seed",
        "architecture",
        "training",
        "epoch",
        "training_iteration",
        "validation_accuracy",
        "validation_loss",
        "train_loss",
        "status",
        "duration_seconds",
        "checkpoint_reference",
        "pruned",
    }
    missing = required - set(trial)
    unknown = set(trial) - required
    if missing or unknown:
        raise ValueError(
            f"invalid trial schema; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if trial["status"] not in {"COMPLETED", "PRUNED", "ERROR"}:
        raise ValueError(f"invalid trial status: {trial['status']}")
    if any("test" in key.lower() for key in trial):
        raise ValueError("test metrics are not allowed in search summaries")


def build_result_document(
    results: Any,
    run_id: str,
    config: ExperimentConfig,
) -> dict[str, Any]:
    document = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at_utc": utc_timestamp(),
        "metric": config.search.metric,
        "mode": config.search.mode,
        "trials": [build_trial_summary(result, run_id, config) for result in results],
    }
    validate_result_document(document)
    return document


def validate_result_document(document: dict[str, Any]) -> None:
    required = {"schema_version", "run_id", "generated_at_utc", "metric", "mode", "trials"}
    if set(document) != required:
        raise ValueError("result document fields do not match schema version 1")
    if document["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported result schema version")
    if document["metric"] != "validation_accuracy" or document["mode"] != "max":
        raise ValueError("result schema supports validation_accuracy maximization only")
    if not isinstance(document["trials"], list):
        raise ValueError("result trials must be a list")
    for trial in document["trials"]:
        validate_trial(trial)
