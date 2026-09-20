from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import make_architecture

from automl_nas.artifacts import (
    build_manifest,
    build_result_document,
    validate_result_document,
)
from automl_nas.cli import main


def test_manifest_contains_required_provenance(smoke_config) -> None:
    manifest = build_manifest(
        smoke_config,
        "smoke-20260101T000000Z-12345678",
        command=["automl-nas", "search", "--config", "configs/smoke.yaml"],
    )
    assert manifest["schema_version"] == 2
    assert manifest["run_id"].startswith("smoke-")
    assert manifest["git"]["available"] is True
    assert len(manifest["git"]["commit_sha"]) == 40
    assert {
        "training",
        "dataset_split",
        "search",
        "candidate_search",
        "confirmation",
        "final_training",
    } == set(manifest["seeds"])
    assert {"python", "torch", "torchvision", "ray", "optuna"} <= set(manifest["software"])
    assert {"os", "logical_cpu_count", "gpu_models", "cuda_available"} <= set(manifest["system"])
    serialized = json.dumps(manifest)
    assert "username" not in serialized.lower()
    assert "hostname" not in serialized.lower()


def test_result_schema_matches_executable_fields(smoke_config, tmp_path: Path) -> None:
    metrics = {
        "trial_id": "abc123",
        "training_iteration": 1,
        "epoch": 1,
        "validation_accuracy": 0.1,
        "validation_loss": 2.3,
        "train_loss": 2.4,
        "time_total_s": 0.5,
        "parameter_count": 8074,
    }
    result = SimpleNamespace(
        metrics=metrics,
        config={
            "architecture": make_architecture(2),
            "training": smoke_config.to_dict()["training"],
        },
        error=None,
        path=str(tmp_path / "candidate_abc123"),
        checkpoint=SimpleNamespace(path=tmp_path / "checkpoint_000000"),
    )
    document = build_result_document([result], "run-123", smoke_config)
    validate_result_document(document)
    trial = document["trials"][0]
    assert trial["status"] == "COMPLETED"
    assert trial["validation_accuracy"] == 0.1
    assert document["history"]
    assert {
        "run_id",
        "trial_id",
        "strategy",
        "search_seed",
        "architecture",
        "epoch",
        "parameter_count",
        "elapsed_trial_seconds",
        "status",
    } <= set(document["history"][0])
    assert document["aggregate"]["epochs_consumed"] == 1
    assert {"started_at_utc", "finished_at_utc", "epochs_consumed", "pruning_iteration"} <= set(
        trial
    )
    assert not any("test" in key for key in trial)


def test_cli_invalid_config_fails_clearly(tmp_path: Path, capsys) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("seed: -1\n", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        main(["validate-config", "--config", str(invalid)])
    assert raised.value.code == 2
    assert "missing keys" in capsys.readouterr().err
