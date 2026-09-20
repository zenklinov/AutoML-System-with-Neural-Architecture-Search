"""Explicit, one-purpose evaluator for already-trained final models."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from automl_nas.artifacts import build_manifest, create_run_id, finalize_manifest, write_json
from automl_nas.config import ExperimentConfig
from automl_nas.data import build_transforms
from automl_nas.models import CandidateCNN
from automl_nas.training import evaluate


def run_locked_test_evaluation(
    config: ExperimentConfig, final_models_path: Path, acknowledged: bool
) -> Path:
    if not acknowledged:
        raise PermissionError("locked test evaluation requires --acknowledge-locked-test")
    document = json.loads(final_models_path.read_text(encoding="utf-8"))
    if document.get("stage") != "final_training":
        raise ValueError("input must be a final-training artifact")
    if config.data.dataset != "cifar10" or config.data.directory is None:
        raise ValueError("locked test evaluation requires CIFAR-10")
    run_id = create_run_id("locked-test")
    root = config.output.root_directory / "runs" / run_id
    manifest_path = root / "manifest.json"
    manifest = build_manifest(
        config,
        run_id,
        entry_point="automl-nas evaluate-locked-test",
        artifacts={"locked_test_summary": "summaries/locked_test.json"},
    )
    write_json(manifest_path, manifest)
    test_dataset = datasets.CIFAR10(
        root=str(config.data.directory),
        train=False,
        download=False,
        transform=build_transforms(config.data)[1],
    )
    loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    device = torch.device("cuda" if config.resources.gpu_per_trial else "cpu")
    criterion = nn.CrossEntropyLoss()
    records = []
    for item in document["records"]:
        model = CandidateCNN(item["architecture"], 3, 10).to(device)
        state = torch.load(item["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        loss, accuracy = evaluate(model, loader, criterion, device)
        records.append(
            {
                "label": item["label"],
                "architecture_id": item["architecture_id"],
                "seed": item["seed"],
                "parameter_count": item["parameter_count"],
                "git_commit_sha": manifest["git"]["commit_sha"],
                "frozen_config_reference": str(config.source_path),
                "test_loss": loss,
                "test_accuracy": accuracy,
            }
        )
    output = root / "summaries" / "locked_test.json"
    write_json(
        output,
        {
            "schema_version": 1,
            "stage": "locked_test_evaluation",
            "run_id": run_id,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "source_final_models": str(final_models_path),
            "records": records,
        },
    )
    finalize_manifest(manifest, manifest_path, "COMPLETED")
    return output
