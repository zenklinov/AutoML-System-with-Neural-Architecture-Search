from __future__ import annotations

from typing import Any

from scripts.analyze_pilot import (
    DISCLAIMER,
    _paired_provenance,
    analyze_calibration,
    analyze_search,
)


def _record(index: int, scores: list[float]) -> dict[str, Any]:
    return {
        "architecture_id": f"architecture-{index}",
        "architecture": {
            "num_blocks": 2,
            "blocks": [
                {
                    "kernel_size": 3,
                    "out_channels": 16,
                    "activation": "relu",
                    "residual": False,
                },
                {
                    "kernel_size": 3,
                    "out_channels": 16,
                    "activation": "relu",
                    "residual": False,
                },
            ],
        },
        "parameter_count": 100 + index,
        "duration_seconds": 10.0,
        "history": [
            {
                "epoch": epoch,
                "validation_accuracy": score,
                "validation_loss": 1.0 - score,
            }
            for epoch, score in enumerate(scores, start=1)
        ],
    }


def test_calibration_analysis_is_descriptive_and_test_isolated() -> None:
    document = {
        "status": "COMPLETED",
        "official_test_access": False,
        "run_id": "calibration-test",
        "records": [
            _record(index, [0.1 + index / 100 + epoch / 100 for epoch in range(12)])
            for index in range(6)
        ],
    }

    report, epoch_rows = analyze_calibration(document)

    assert report["disclaimer"] == DISCLAIMER
    assert report["official_test_access"] is False
    assert len(report["hypothetical_successive_halving"]) == 5
    assert len(epoch_rows) == 6 * 12
    assert {row["epoch_2_rank"] for row in report["architectures"]} == {
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    }


def test_calibration_analysis_rejects_unfinished_input() -> None:
    document = {
        "status": "RUNNING",
        "official_test_access": False,
        "run_id": "calibration-test",
        "records": [],
    }

    try:
        analyze_calibration(document)
    except ValueError as error:
        assert "complete" in str(error)
    else:
        raise AssertionError("unfinished calibration should be rejected")


def test_search_analysis_classifies_tpe_startup_and_adaptive_trials() -> None:
    trials = []
    for index in range(12):
        trials.append(
            {
                "trial_id": f"trial-{index}",
                "search_strategy": "bayesian",
                "architecture": _record(index, [0.5])["architecture"],
                "parameter_count": 100 + index,
                "status": "COMPLETED",
                "epochs_consumed": 12,
                "pruning_iteration": None,
                "validation_accuracy": 0.5 + index / 100,
                "validation_loss": 0.5,
                "train_loss": 0.4,
                "duration_seconds": 10.0 + index,
            }
        )
    document = {
        "run_id": "tpe-test",
        "trials": trials,
        "aggregate": {
            "completed_count": 12,
            "pruned_count": 0,
            "failed_count": 0,
            "epochs_consumed": 144,
            "search_duration_seconds": 200.0,
            "cpu_hours": 0.1,
            "gpu_hours": 0.0,
        },
    }

    report, rows = analyze_search(document)

    assert report["trial_count"] == 12
    assert [row["proposal_phase"] for row in rows[:10]] == ["startup"] * 10
    assert [row["proposal_phase"] for row in rows[10:]] == ["adaptive"] * 2
    assert report["best_validation_accuracy"] == 0.61


def test_search_analysis_summarizes_pruning_and_capacity() -> None:
    completed = {
        "trial_id": "completed",
        "search_strategy": "random",
        "architecture": _record(1, [0.5])["architecture"],
        "parameter_count": 200,
        "status": "COMPLETED",
        "epochs_consumed": 12,
        "pruning_iteration": None,
        "validation_accuracy": 0.8,
        "validation_loss": 0.2,
        "train_loss": 0.1,
        "duration_seconds": 12.0,
    }
    pruned = {
        **completed,
        "trial_id": "pruned",
        "parameter_count": 100,
        "status": "PRUNED",
        "epochs_consumed": 4,
        "pruning_iteration": 4,
        "validation_accuracy": 0.5,
        "duration_seconds": 4.0,
    }
    history = [
        {
            "trial_id": "completed",
            "epoch": epoch,
            "validation_accuracy": 0.5 + epoch / 40,
        }
        for epoch in range(1, 13)
    ]
    document = {
        "run_id": "random-test",
        "trials": [completed, pruned],
        "history": history,
        "aggregate": {
            "completed_count": 1,
            "pruned_count": 1,
            "failed_count": 0,
            "epochs_consumed": 16,
            "search_duration_seconds": 16.0,
            "cpu_hours": 0.0,
            "gpu_hours": 1.0,
        },
    }

    report, _ = analyze_search(document)

    assert report["pruning_iterations"] == {"4": 1}
    assert abs(report["capacity"]["terminal_score_parameter_spearman"] - 1.0) < 1e-12
    assert abs(report["late_epoch_learning"]["median_gain_epoch_8_to_final"] - 0.1) < 1e-12


def test_paired_provenance_rejects_non_strategy_config_difference() -> None:
    base = {
        "status": "COMPLETED",
        "git": {"commit_sha": "abc", "dirty": False},
        "config": {
            "search": {"strategy": "random", "num_trials": 16},
            "output": {"run_label": "random"},
        },
        "software": {},
        "system": {},
        "execution": {"search_strategy": "random", "gpu_per_trial": 1},
        "run_id": "random",
        "started_at_utc": "start",
        "finished_at_utc": "finish",
    }
    tpe = {
        **base,
        "config": {
            "search": {"strategy": "bayesian", "num_trials": 20},
            "output": {"run_label": "tpe"},
        },
    }

    try:
        _paired_provenance(base, tpe)
    except ValueError as error:
        assert "differ" in str(error)
    else:
        raise AssertionError("mismatched paired configs should be rejected")
