from __future__ import annotations

from typing import Any

from scripts.analyze_pilot import DISCLAIMER, analyze_calibration, analyze_search


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
