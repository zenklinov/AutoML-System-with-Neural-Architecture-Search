from __future__ import annotations

import json
from pathlib import Path

import pytest

from automl_nas.search import run_search


@pytest.mark.smoke
def test_two_trial_cpu_search_creates_manifest_and_summary(smoke_config) -> None:
    outcome = run_search(
        smoke_config,
        command=["automl-nas", "search", "--config", "configs/smoke.yaml"],
    )
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    results = json.loads(outcome.result_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    assert manifest["run_id"] == outcome.run_id
    assert results["run_id"] == outcome.run_id
    assert len(results["trials"]) == 2
    assert all(trial["status"] == "COMPLETED" for trial in results["trials"])
    checkpoint_paths = [Path(trial["checkpoint_reference"]) for trial in results["trials"]]
    assert all(path.is_dir() for path in checkpoint_paths)
    assert all((path / "training_state.pt").is_file() for path in checkpoint_paths)
    assert all("test" not in key for trial in results["trials"] for key in trial)
