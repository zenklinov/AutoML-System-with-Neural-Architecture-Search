#!/usr/bin/env python3
"""Resumable WSL-only orchestration for the frozen pre-test benchmark."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "final_runner"
STATE_PATH = ARTIFACT_ROOT / "state.json"
LOG_ROOT = ARTIFACT_ROOT / "logs"
FREEZE_PATH = ROOT / "results" / "final" / "pretest_freeze.json"

CONFIGS = {
    "random": ROOT / "configs" / "final_random_search.yaml",
    "bayesian": ROOT / "configs" / "final_tpe_search.yaml",
    "confirmation": ROOT / "configs" / "final_confirmation.yaml",
    "final_training": ROOT / "configs" / "final_training.yaml",
}

STAGES = (
    "preflight",
    "random_2026",
    "tpe_2026",
    "tpe_2027",
    "random_2027",
    "random_2028",
    "tpe_2028",
    "validate_searches",
    "aggregate_searches",
    "confirmation",
    "select_winners",
    "final_training",
    "pretest_freeze",
)

SEARCH_STAGES = {
    "random_2026": ("random", 2026),
    "tpe_2026": ("bayesian", 2026),
    "tpe_2027": ("bayesian", 2027),
    "random_2027": ("random", 2027),
    "random_2028": ("random", 2028),
    "tpe_2028": ("bayesian", 2028),
}

LOG_NAMES = {
    **{stage: f"{stage}.log" for stage in SEARCH_STAGES},
    "preflight": "preflight.log",
    "validate_searches": "validation.log",
    "aggregate_searches": "aggregation.log",
    "confirmation": "confirmation.log",
    "select_winners": "confirmation.log",
    "final_training": "final_training.log",
    "pretest_freeze": "pretest_freeze.log",
}

EXPECTED_PACKAGES = {
    "torch": "2.4.0",
    "torchvision": "0.19.0",
    "ray": "2.40.0",
    "optuna": "4.0.0",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_sha() -> str:
    return git("rev-parse", "HEAD")


def assert_clean_git() -> None:
    if git("status", "--porcelain"):
        raise RuntimeError("working tree is dirty; commit or remove changes before benchmarking")
    if git("branch", "--show-current") != "final-benchmark":
        raise RuntimeError("benchmark must run from the final-benchmark branch")


def config_hashes() -> dict[str, str]:
    return {name: sha256(path) for name, path in CONFIGS.items()}


def environment_identity() -> dict[str, Any]:
    import torch

    packages = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    distributions = sorted(
        f"{distribution.metadata['Name'].lower()}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    nvidia_smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    return {
        "platform": platform.system(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": packages,
        "installed_packages_sha256": hashlib.sha256(
            "\n".join(distributions).encode("utf-8")
        ).hexdigest(),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "nvidia_smi_identity": nvidia_smi,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def validate_environment(identity: dict[str, Any]) -> None:
    if identity["platform"] != "Linux":
        raise RuntimeError("final benchmark is restricted to Linux/WSL")
    if not identity["python"].startswith("3.11."):
        raise RuntimeError(f"Python 3.11 is required, found {identity['python']}")
    for name, expected in EXPECTED_PACKAGES.items():
        actual = str(identity["packages"][name]).split("+")[0]
        if actual != expected:
            raise RuntimeError(f"{name} must be {expected}, found {identity['packages'][name]}")
    if identity["cuda_runtime"] != "12.4":
        raise RuntimeError(f"PyTorch CUDA runtime must be 12.4, found {identity['cuda_runtime']}")
    if not identity["cuda_available"] or len(identity["gpu_names"]) != 1:
        raise RuntimeError("exactly one CUDA GPU must be visible")
    if "RTX 4060" not in identity["gpu_names"][0]:
        raise RuntimeError(f"expected RTX 4060 GPU, found {identity['gpu_names'][0]}")
    if "RTX 4060" not in identity["nvidia_smi_identity"]:
        raise RuntimeError("nvidia-smi did not report the expected RTX 4060 GPU")
    if identity["cublas_workspace_config"] != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")


def validate_protocol() -> None:
    from automl_nas.config import load_config

    for path in CONFIGS.values():
        config = load_config(path)
        if not config.protocol.final_budget_locked:
            raise RuntimeError(f"final budget is not locked in {path}")
        expected = {
            "candidate_search_seeds": (2026, 2027, 2028),
            "search_training_seed": 4242,
            "confirmation_seeds": (3101, 3102, 3103),
            "final_training_seeds": (4101, 4102, 4103, 4104, 4105),
        }
        for field, value in expected.items():
            if getattr(config.protocol, field) != value:
                raise RuntimeError(f"frozen protocol mismatch: {path.name}:{field}")
        if config.training.max_epochs != 20:
            raise RuntimeError(f"frozen max_epochs mismatch in {path.name}")
        if config.resources.gpu_per_trial != 1:
            raise RuntimeError(f"one GPU is required in {path.name}")
    for strategy in ("random", "bayesian"):
        config = load_config(CONFIGS[strategy])
        if (
            config.search.strategy != strategy
            or config.search.num_trials != 20
            or config.search.grace_period_epochs != 4
            or config.search.reduction_factor != 2
            or config.search.max_concurrent_trials != 1
        ):
            raise RuntimeError(f"frozen search mismatch in {CONFIGS[strategy].name}")


def new_state(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "RUNNING",
        "git_sha": git_sha(),
        "config_hashes": config_hashes(),
        "environment_identity": identity,
        "current_stage": None,
        "current_strategy": None,
        "current_seed": None,
        "completed_stages": [],
        "valid_runs": {},
        "recoverable_runs": {},
        "aborted_runs": [],
        "stage_history": [],
        "failure_reason": None,
        "official_test_access": False,
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
    }


def assert_invariants(state: dict[str, Any], identity: dict[str, Any]) -> None:
    if state["git_sha"] != git_sha():
        raise RuntimeError("Git SHA changed since the benchmark began")
    if state["config_hashes"] != config_hashes():
        raise RuntimeError("a frozen configuration hash changed")
    if state["environment_identity"] != identity:
        raise RuntimeError("the dependency or GPU environment changed")
    if state.get("official_test_access") is not False:
        raise RuntimeError("official-test access flag is not false")
    assert_clean_git()


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def validate_search_result(
    result_path: Path, strategy: str, seed: int, expected_sha: str
) -> dict[str, Any]:
    from automl_nas.artifacts import validate_result_document

    document = read_json(result_path)
    validate_result_document(document)
    aggregate = document["aggregate"]
    if aggregate["trial_count"] != 20:
        raise RuntimeError(f"{strategy}-{seed} has {aggregate['trial_count']} trials, not 20")
    if aggregate["failed_count"] != 0:
        raise RuntimeError(f"{strategy}-{seed} contains a failed trial")
    if aggregate["completed_count"] + aggregate["pruned_count"] != 20:
        raise RuntimeError(f"{strategy}-{seed} has an incomplete trial lifecycle")
    for trial in document["trials"]:
        if trial["search_strategy"] != strategy or trial["search_seed"] != seed:
            raise RuntimeError(f"{strategy}-{seed} contains mismatched trial provenance")
        if trial["training_seed"] != 4242:
            raise RuntimeError(f"{strategy}-{seed} contains a wrong training seed")
    manifest = read_json(result_path.parents[1] / "manifest.json")
    if manifest["status"] != "COMPLETED":
        raise RuntimeError(f"{strategy}-{seed} manifest is not complete")
    if manifest["git"]["commit_sha"] != expected_sha or manifest["git"]["dirty"]:
        raise RuntimeError(f"{strategy}-{seed} Git provenance is invalid")
    if manifest["system"]["os"] != "Linux":
        raise RuntimeError(f"{strategy}-{seed} is not a Linux run")
    if any("test" in key.lower() for key in document):
        raise RuntimeError("search summary contains a test field")
    return document


def matching_resume_directory(config: Any, state: dict[str, Any]) -> Path | None:
    runs_root = config.output.root_directory / "runs"
    if not runs_root.exists():
        return None
    expected_config = config.to_dict()
    candidates = sorted(
        runs_root.glob("*/manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True
    )
    recorded = state.setdefault("recoverable_runs", {}).get(state["current_stage"])
    if recorded:
        recorded_manifest = runs_root / recorded / "manifest.json"
        candidates = [
            recorded_manifest,
            *[path for path in candidates if path != recorded_manifest],
        ]
    for manifest_path in candidates:
        if manifest_path.parent.name in state["aborted_runs"]:
            continue
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("status") in {"RUNNING", "FAILED"}
            and manifest.get("config") == expected_config
            and manifest.get("git", {}).get("commit_sha") == state["git_sha"]
            and manifest.get("git", {}).get("dirty") is False
            and manifest.get("system", {}).get("os") == "Linux"
            and manifest.get("software")
            == {
                "python": state["environment_identity"]["python"],
                "torch": state["environment_identity"]["packages"]["torch"],
                "torchvision": state["environment_identity"]["packages"]["torchvision"],
                "ray": state["environment_identity"]["packages"]["ray"],
                "optuna": state["environment_identity"]["packages"]["optuna"],
            }
            and (manifest_path.parent / "ray" / "tune").exists()
        ):
            return manifest_path.parent
        if recorded and manifest_path == runs_root / recorded / "manifest.json":
            state["aborted_runs"].append(recorded)
            state["recoverable_runs"].pop(state["current_stage"], None)
            write_json(STATE_PATH, state)
    return None


def run_search_stage(stage: str, state: dict[str, Any]) -> None:
    from automl_nas.config import load_config
    from automl_nas.search import run_search

    strategy, seed = SEARCH_STAGES[stage]
    base = load_config(CONFIGS[strategy])
    config = replace(base, search=replace(base.search, seed=seed))
    resume_directory = matching_resume_directory(config, state)
    if resume_directory is not None:
        print(f"Resuming validated Ray state: {resume_directory}")
    before = set(config.output.root_directory.glob("runs/*"))
    try:
        outcome = run_search(
            config,
            command=["scripts/run_final_benchmark.sh", f"stage={stage}"],
            resume_run_directory=resume_directory,
        )
    except BaseException:
        after = set(config.output.root_directory.glob("runs/*"))
        candidates = [resume_directory] if resume_directory else sorted(after - before)
        for candidate in candidates:
            if candidate is not None:
                state.setdefault("recoverable_runs", {})[stage] = candidate.name
        write_json(STATE_PATH, state)
        raise
    validate_search_result(outcome.result_path, strategy, seed, state["git_sha"])
    state["valid_runs"][stage] = {
        "run_id": outcome.run_id,
        "result_path": str(outcome.result_path),
        "strategy": strategy,
        "seed": seed,
    }
    state.setdefault("recoverable_runs", {}).pop(stage, None)


def all_result_paths(state: dict[str, Any]) -> list[Path]:
    return [Path(state["valid_runs"][stage]["result_path"]) for stage in SEARCH_STAGES]


def validate_all_searches(state: dict[str, Any]) -> None:
    for stage, (strategy, seed) in SEARCH_STAGES.items():
        record = state["valid_runs"].get(stage)
        if record is None:
            raise RuntimeError(f"missing run record for {stage}")
        validate_search_result(Path(record["result_path"]), strategy, seed, state["git_sha"])
    print("All six final searches have complete Linux provenance and 20 valid lifecycles.")


def prove_completed_stage(stage: str, state: dict[str, Any]) -> None:
    """Re-prove persisted completion before a restarted runner skips a stage."""
    if stage == "preflight":
        log = LOG_ROOT / LOG_NAMES[stage]
        if not log.is_file() or '"status": "PASSED"' not in log.read_text(encoding="utf-8"):
            raise RuntimeError("preflight log does not contain a passing CUDA/Ray result")
        return
    if stage in SEARCH_STAGES:
        strategy, seed = SEARCH_STAGES[stage]
        record = state["valid_runs"].get(stage)
        if record is None:
            raise RuntimeError(f"missing persisted run for {stage}")
        validate_search_result(Path(record["result_path"]), strategy, seed, state["git_sha"])
        return
    if stage == "validate_searches":
        validate_all_searches(state)
        return
    if stage == "aggregate_searches":
        record = state["valid_runs"].get("aggregation")
        if record is None:
            raise RuntimeError("missing aggregation record")
        document = read_json(Path(record["summary_path"]))
        if (
            document.get("official_test_access") is not False
            or document.get("git_sha") != state["git_sha"]
            or document.get("config_hashes") != state["config_hashes"]
            or document.get("totals", {}).get("trials") != 120
            or document.get("totals", {}).get("failed") != 0
        ):
            raise RuntimeError("aggregation summary is incomplete or invalid")
        return
    if stage in {"confirmation", "select_winners"}:
        record = state["valid_runs"].get("confirmation")
        if record is None:
            raise RuntimeError("missing confirmation record")
        document = read_json(Path(record["summary_path"]))
        if (
            document.get("official_test_access") is not False
            or document.get("seeds") != [3101, 3102, 3103]
            or set(document.get("strategy_winners", {})) != {"random", "bayesian"}
        ):
            raise RuntimeError("confirmation summary is incomplete or invalid")
        if stage == "select_winners" and "selection" not in state["valid_runs"]:
            raise RuntimeError("winner-selection record is missing")
        return
    if stage == "final_training":
        record = state["valid_runs"].get("final_training")
        if record is None:
            raise RuntimeError("missing final-training record")
        document = read_json(Path(record["summary_path"]))
        if document.get("official_test_access") is not False:
            raise RuntimeError("final-training official-test flag is invalid")
        if {row["seed"] for row in document.get("records", [])} != {
            4101,
            4102,
            4103,
            4104,
            4105,
        }:
            raise RuntimeError("final-training seeds are incomplete")
        if any(not Path(row["checkpoint"]).is_file() for row in document["records"]):
            raise RuntimeError("a final-training checkpoint is missing")
        return
    if stage == "pretest_freeze":
        document = read_json(FREEZE_PATH)
        if (
            document.get("stage") != "PRETEST_FREEZE_COMPLETE"
            or document.get("official_test_access") is not False
            or document.get("git_sha") != state["git_sha"]
            or document.get("config_hashes") != state["config_hashes"]
        ):
            raise RuntimeError("pre-test freeze artifact is incomplete or invalid")
        return
    raise RuntimeError(f"cannot prove unknown stage: {stage}")


def invalidate_from(stage: str, state: dict[str, Any], reason: str) -> None:
    """Invalidate one stage and every dependent stage after failed re-validation."""
    index = STAGES.index(stage)
    invalid_stages = set(STAGES[index:])
    record_keys = invalid_stages | {
        "aggregation" if "aggregate_searches" in invalid_stages else "",
        "selection" if "select_winners" in invalid_stages else "",
        "final_training" if "final_training" in invalid_stages else "",
        "pretest_freeze" if "pretest_freeze" in invalid_stages else "",
    }
    if "confirmation" in invalid_stages:
        record_keys.add("confirmation")
    for key in record_keys - {""}:
        record = state["valid_runs"].pop(key, None)
        if record and record.get("run_id"):
            state["aborted_runs"].append(record["run_id"])
        recoverable = state.setdefault("recoverable_runs", {}).pop(key, None)
        if recoverable:
            state["aborted_runs"].append(recoverable)
    state["completed_stages"] = [
        completed for completed in state["completed_stages"] if completed not in invalid_stages
    ]
    state["stage_history"].append(
        {
            "stage": stage,
            "status": "INVALIDATED",
            "reason": reason,
            "started_at_utc": utc_now(),
            "finished_at_utc": utc_now(),
        }
    )
    write_json(STATE_PATH, state)


def aggregate_searches(state: dict[str, Any]) -> None:
    runs = []
    totals = {"trials": 0, "completed": 0, "pruned": 0, "failed": 0}
    for stage, result_path in zip(SEARCH_STAGES, all_result_paths(state), strict=True):
        document = read_json(result_path)
        aggregate = document["aggregate"]
        row = {
            "stage": stage,
            "run_id": document["run_id"],
            "result_path": str(result_path),
            "aggregate": aggregate,
        }
        runs.append(row)
        totals["trials"] += aggregate["trial_count"]
        totals["completed"] += aggregate["completed_count"]
        totals["pruned"] += aggregate["pruned_count"]
        totals["failed"] += aggregate["failed_count"]
    output = ARTIFACT_ROOT / "final_search_summary.json"
    write_json(
        output,
        {
            "schema_version": 1,
            "stage": "final_search_aggregation",
            "generated_at_utc": utc_now(),
            "git_sha": state["git_sha"],
            "config_hashes": state["config_hashes"],
            "runs": runs,
            "totals": totals,
            "official_test_access": False,
        },
    )
    state["valid_runs"]["aggregation"] = {"summary_path": str(output)}
    print(output)


def run_confirmation_stage(state: dict[str, Any]) -> None:
    from automl_nas.config import load_config
    from automl_nas.workflows import run_confirmation

    output = run_confirmation(load_config(CONFIGS["confirmation"]), all_result_paths(state), 3)
    document = read_json(output)
    if document.get("official_test_access") is not False:
        raise RuntimeError("confirmation official-test flag is invalid")
    if document.get("seeds") != [3101, 3102, 3103]:
        raise RuntimeError("confirmation seeds are invalid")
    if set(document.get("strategy_winners", {})) != {"random", "bayesian"}:
        raise RuntimeError("confirmation did not produce Random and TPE winners")
    state["valid_runs"]["confirmation"] = {
        "run_id": document["run_id"],
        "summary_path": str(output),
    }


def select_winners(state: dict[str, Any]) -> None:
    confirmation = read_json(Path(state["valid_runs"]["confirmation"]["summary_path"]))
    winners = confirmation["strategy_winners"]
    if set(winners) != {"random", "bayesian"}:
        raise RuntimeError("Random and TPE winners are both required")
    state["valid_runs"]["selection"] = {
        "random_architecture_id": winners["random"]["architecture_id"],
        "tpe_architecture_id": winners["bayesian"]["architecture_id"],
    }
    print(json.dumps(state["valid_runs"]["selection"], indent=2))


def run_final_training_stage(state: dict[str, Any]) -> None:
    from automl_nas.config import load_config
    from automl_nas.workflows import run_final_training

    confirmation_path = Path(state["valid_runs"]["confirmation"]["summary_path"])
    output = run_final_training(load_config(CONFIGS["final_training"]), confirmation_path)
    document = read_json(output)
    if document.get("official_test_access") is not False:
        raise RuntimeError("final-training official-test flag is invalid")
    expected_seeds = {4101, 4102, 4103, 4104, 4105}
    if {record["seed"] for record in document["records"]} != expected_seeds:
        raise RuntimeError("final-training seeds are incomplete")
    if any(not Path(record["checkpoint"]).is_file() for record in document["records"]):
        raise RuntimeError("a final-training checkpoint is missing")
    state["valid_runs"]["final_training"] = {
        "run_id": document["run_id"],
        "summary_path": str(output),
    }


def create_pretest_freeze(state: dict[str, Any]) -> None:
    confirmation_path = Path(state["valid_runs"]["confirmation"]["summary_path"])
    training_path = Path(state["valid_runs"]["final_training"]["summary_path"])
    confirmation = read_json(confirmation_path)
    training = read_json(training_path)
    write_json(
        FREEZE_PATH,
        {
            "schema_version": 1,
            "stage": "PRETEST_FREEZE_COMPLETE",
            "generated_at_utc": utc_now(),
            "git_sha": state["git_sha"],
            "config_hashes": state["config_hashes"],
            "selected_random_architecture": confirmation["strategy_winners"]["random"],
            "selected_tpe_architecture": confirmation["strategy_winners"]["bayesian"],
            "baseline": next(
                row
                for row in confirmation["ranked_candidates"]
                if "fixed_reference_baseline" in row["source_strategies"]
            ),
            "confirmation_summary": str(confirmation_path),
            "confirmation_seeds": confirmation["seeds"],
            "final_training_summary": str(training_path),
            "final_training_seeds": [4101, 4102, 4103, 4104, 4105],
            "checkpoint_references": [record["checkpoint"] for record in training["records"]],
            "official_test_access": False,
        },
    )
    state["valid_runs"]["pretest_freeze"] = {"path": str(FREEZE_PATH)}
    print(FREEZE_PATH)


def run_preflight() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gpu_smoke.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)


def execute_stage(stage: str, state: dict[str, Any]) -> None:
    if stage in SEARCH_STAGES:
        run_search_stage(stage, state)
    elif stage == "preflight":
        run_preflight()
    elif stage == "validate_searches":
        validate_all_searches(state)
    elif stage == "aggregate_searches":
        aggregate_searches(state)
    elif stage == "confirmation":
        run_confirmation_stage(state)
    elif stage == "select_winners":
        select_winners(state)
    elif stage == "final_training":
        run_final_training_stage(state)
    elif stage == "pretest_freeze":
        create_pretest_freeze(state)
    else:
        raise RuntimeError(f"unknown stage: {stage}")


def run() -> None:
    assert_clean_git()
    validate_protocol()
    identity = environment_identity()
    validate_environment(identity)
    state = read_json(STATE_PATH) if STATE_PATH.exists() else new_state(identity)
    assert_invariants(state, identity)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for completed in tuple(state["completed_stages"]):
        try:
            prove_completed_stage(completed, state)
        except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as error:
            invalidate_from(completed, state, f"{type(error).__name__}: {error}")
            print(f"INVALIDATED {completed}: {error}")
            break
    for stage in STAGES:
        if stage in state["completed_stages"]:
            print(f"SKIP validated stage: {stage}")
            continue
        current_identity = environment_identity()
        validate_environment(current_identity)
        assert_invariants(state, current_identity)
        strategy, seed = SEARCH_STAGES.get(stage, (None, None))
        state.update(
            {
                "status": "RUNNING",
                "current_stage": stage,
                "current_strategy": strategy,
                "current_seed": seed,
                "failure_reason": None,
            }
        )
        event = {"stage": stage, "started_at_utc": utc_now(), "finished_at_utc": None}
        state["stage_history"].append(event)
        write_json(STATE_PATH, state)
        log_path = LOG_ROOT / LOG_NAMES[stage]
        runs_before = set((ROOT / "artifacts" / "runs").glob("*"))
        try:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{utc_now()}] START {stage}\n")
                tee_out, tee_err = Tee(sys.stdout, log), Tee(sys.stderr, log)
                with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                    execute_stage(stage, state)
                log.write(f"[{utc_now()}] COMPLETE {stage}\n")
        except BaseException as error:
            runs_after = set((ROOT / "artifacts" / "runs").glob("*"))
            if stage not in SEARCH_STAGES:
                for directory in sorted(runs_after - runs_before):
                    if directory.name not in state["aborted_runs"]:
                        state["aborted_runs"].append(directory.name)
            event["finished_at_utc"] = utc_now()
            event["status"] = "FAILED"
            state["status"] = "FAILED"
            state["failure_reason"] = f"{type(error).__name__}: {error}"
            write_json(STATE_PATH, state)
            raise
        event["finished_at_utc"] = utc_now()
        event["status"] = "COMPLETED"
        state["completed_stages"].append(stage)
        write_json(STATE_PATH, state)
    state.update(
        {
            "status": "PRETEST_FREEZE_COMPLETE",
            "current_stage": None,
            "current_strategy": None,
            "current_seed": None,
            "failure_reason": None,
            "finished_at_utc": utc_now(),
            "official_test_access": False,
        }
    )
    write_json(STATE_PATH, state)
    print("PRETEST_FREEZE_COMPLETE")
    print("Official CIFAR-10 test evaluation was not run.")


def dry_run() -> None:
    validate_protocol()
    identity = environment_identity()
    validate_environment(identity)
    run_preflight()
    print("DRY_RUN_PASSED")
    print(f"Repository: {ROOT}")
    print(f"Git SHA: {git_sha()}")
    print(f"Working tree clean: {not bool(git('status', '--porcelain'))}")
    print(f"State: {STATE_PATH}")
    print(f"Logs: {LOG_ROOT}")
    print(f"Pre-test freeze: {FREEZE_PATH}")
    print("Stage plan:")
    for index, stage in enumerate(STAGES, 1):
        print(f"  {index:02d}. {stage}")
    print("Official-test command present in plan: false")
    print("No search, confirmation, final training, or test evaluation was started.")


def ray_progress(state: dict[str, Any]) -> dict[str, Any]:
    stage = state.get("current_stage")
    if stage not in SEARCH_STAGES:
        return {}
    strategy, seed = SEARCH_STAGES[stage]
    runs_root = ROOT / "artifacts" / "runs"
    matching = sorted(
        runs_root.glob("final-*-search-*/manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for manifest_path in matching:
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("execution", {}).get("search_strategy") == strategy
            and manifest.get("seeds", {}).get("search") == seed
            and manifest.get("git", {}).get("commit_sha") == state.get("git_sha")
        ):
            result_files = list(manifest_path.parent.glob("ray/tune/**/result.json"))
            rows = []
            for result_file in result_files:
                try:
                    lines = result_file.read_text(encoding="utf-8").splitlines()
                    if lines:
                        rows.append(json.loads(lines[-1]))
                except (OSError, json.JSONDecodeError):
                    continue
            return {
                "active_trial": max(
                    (str(row.get("trial_id", "unknown")) for row in rows), default="unknown"
                ),
                "trial_count": len(rows),
                "latest_epoch": max(
                    (int(row.get("training_iteration", row.get("epoch", 0))) for row in rows),
                    default=0,
                ),
            }
    return {}


def status() -> None:
    if not STATE_PATH.exists():
        state: dict[str, Any] = {
            "status": "NOT_STARTED",
            "current_stage": None,
            "current_strategy": None,
            "current_seed": None,
            "completed_stages": [],
            "valid_runs": {},
            "failure_reason": None,
            "official_test_access": False,
            "git_sha": git_sha(),
            "started_at_utc": None,
        }
    else:
        state = read_json(STATE_PATH)
    progress = ray_progress(state)
    counts = {"trial_count": 0, "completed": 0, "pruned": 0, "failed": 0}
    stage = state.get("current_stage")
    if stage in state.get("valid_runs", {}):
        result_path = Path(state["valid_runs"][stage]["result_path"])
        if result_path.exists():
            aggregate = read_json(result_path)["aggregate"]
            counts = {
                "trial_count": aggregate["trial_count"],
                "completed": aggregate["completed_count"],
                "pruned": aggregate["pruned_count"],
                "failed": aggregate["failed_count"],
            }
    elapsed = None
    if state.get("started_at_utc"):
        elapsed = int(time.time() - datetime.fromisoformat(state["started_at_utc"]).timestamp())
    print(f"benchmark_status: {state['status']}")
    print(f"current_stage: {state.get('current_stage') or '-'}")
    print(f"current_strategy: {state.get('current_strategy') or '-'}")
    print(f"current_seed: {state.get('current_seed') or '-'}")
    print(f"completed_stages: {len(state.get('completed_stages', []))}/{len(STAGES)}")
    print(f"active_trial: {progress.get('active_trial', '-')}")
    print(f"trial_count: {progress.get('trial_count', counts['trial_count'])}")
    print(f"completed: {counts['completed']}")
    print(f"pruned: {counts['pruned']}")
    print(f"failed: {counts['failed']}")
    print(f"latest_epoch: {progress.get('latest_epoch', '-')}")
    print(f"elapsed_seconds: {elapsed if elapsed is not None else '-'}")
    print(f"git_sha: {state.get('git_sha') or '-'}")
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "unavailable"
    print(f"gpu_utilization_memory: {gpu}")
    print(f"official_test_access: {str(state.get('official_test_access', False)).lower()}")
    print(f"failure_reason: {state.get('failure_reason') or '-'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "dry-run", "status"))
    arguments = parser.parse_args()
    if arguments.command == "run":
        run()
    elif arguments.command == "dry-run":
        dry_run()
    else:
        status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
