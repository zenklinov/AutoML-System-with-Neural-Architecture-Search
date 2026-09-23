from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "final_benchmark.py"


def _module():
    spec = importlib.util.spec_from_file_location("final_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_order_is_frozen_and_ends_before_official_test() -> None:
    module = _module()
    assert module.STAGES == (
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
    assert module.SEARCH_STAGES == {
        "random_2026": ("random", 2026),
        "tpe_2026": ("bayesian", 2026),
        "tpe_2027": ("bayesian", 2027),
        "random_2027": ("random", 2027),
        "random_2028": ("random", 2028),
        "tpe_2028": ("bayesian", 2028),
    }
    assert "official_test" not in module.STAGES
    assert "locked_test" not in module.STAGES


def test_protocol_configs_remain_frozen() -> None:
    module = _module()
    module.validate_protocol()


def test_new_state_is_explicitly_pretest_only(monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "git_sha", lambda: "a" * 40)
    monkeypatch.setattr(module, "config_hashes", lambda: {"config": "hash"})
    state = module.new_state({"platform": "Linux"})
    assert state["official_test_access"] is False
    assert state["completed_stages"] == []
    assert state["valid_runs"] == {}
    assert state["recoverable_runs"] == {}
    assert state["aborted_runs"] == []


def test_invalid_stage_discards_every_dependent_completion(tmp_path, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "STATE_PATH", tmp_path / "state.json")
    state = {
        "completed_stages": list(module.STAGES),
        "valid_runs": {
            "random_2026": {"run_id": "bad-search"},
            "confirmation": {"run_id": "dependent-confirmation"},
            "final_training": {"run_id": "dependent-training"},
        },
        "recoverable_runs": {"tpe_2026": "interrupted-tpe"},
        "aborted_runs": [],
        "stage_history": [],
    }
    module.invalidate_from("random_2026", state, "corrupt artifact")
    assert state["completed_stages"] == ["preflight"]
    assert state["valid_runs"] == {}
    assert set(state["aborted_runs"]) == {
        "bad-search",
        "dependent-confirmation",
        "dependent-training",
        "interrupted-tpe",
    }
    assert state["stage_history"][-1]["status"] == "INVALIDATED"


def test_status_before_start_is_read_only(tmp_path, monkeypatch, capsys) -> None:
    module = _module()
    missing = tmp_path / "state.json"
    monkeypatch.setattr(module, "STATE_PATH", missing)
    monkeypatch.setattr(module, "git_sha", lambda: "b" * 40)

    def unavailable(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(module.subprocess, "run", unavailable)
    module.status()
    output = capsys.readouterr().out
    assert "benchmark_status: NOT_STARTED" in output
    assert "official_test_access: false" in output
    assert not missing.exists()


def test_shell_runner_has_no_locked_test_entrypoint() -> None:
    source = (ROOT / "scripts" / "run_final_benchmark.sh").read_text(encoding="utf-8")
    helper = SCRIPT.read_text(encoding="utf-8")
    forbidden = "evaluate" + "-locked-test"
    assert forbidden not in source
    assert forbidden not in helper
