"""Create compact, descriptive calibration and comparative-pilot evidence.

The analysis intentionally avoids hypothesis tests.  It reads only validation
artifacts produced by the calibration/search workflows and never loads CIFAR-10.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DISCLAIMER = "PILOT / DIAGNOSTIC — NOT FINAL BENCHMARK"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ranks(values: list[float], *, reverse: bool = True) -> list[float]:
    """Return one-based average ranks, with the best value ranked first."""
    order = sorted(range(len(values)), key=values.__getitem__, reverse=reverse)
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2
        for index in order[cursor:end]:
            result[index] = average_rank
        cursor = end
    return result


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=False))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def _spearman(values: list[float], final_values: list[float]) -> float | None:
    return _pearson(_ranks(values), _ranks(final_values))


def _architecture_traits(architecture: dict[str, Any]) -> dict[str, Any]:
    blocks = architecture["blocks"]
    return {
        "blocks": architecture["num_blocks"],
        "has_residual": any(block["residual"] for block in blocks),
        "all_residual": all(block["residual"] for block in blocks),
        "has_silu": any(block["activation"] == "silu" for block in blocks),
        "all_silu": all(block["activation"] == "silu" for block in blocks),
    }


def _successive_halving_risk(
    records: list[dict[str, Any]], grace_period: int, max_epoch: int
) -> dict[str, Any]:
    """Approximate synchronous halving; real ASHA is asynchronous and timing-sensitive."""
    active = list(range(len(records)))
    rounds: list[dict[str, Any]] = []
    milestone = grace_period
    while milestone < max_epoch and len(active) > 1:
        scores = {
            index: float(records[index]["history"][milestone - 1]["validation_accuracy"])
            for index in active
        }
        keep_count = max(1, math.ceil(len(active) / 2))
        ordered = sorted(active, key=lambda index: scores[index], reverse=True)
        kept = ordered[:keep_count]
        removed = ordered[keep_count:]
        rounds.append(
            {
                "epoch": milestone,
                "active_architecture_ids": [records[index]["architecture_id"] for index in active],
                "retained_architecture_ids": [records[index]["architecture_id"] for index in kept],
                "at_risk_architecture_ids": [
                    records[index]["architecture_id"] for index in removed
                ],
            }
        )
        active = kept
        milestone *= 2
    return {
        "grace_period": grace_period,
        "method": "synchronous successive-halving approximation; actual ASHA may differ",
        "rounds": rounds,
        "surviving_architecture_ids": [records[index]["architecture_id"] for index in active],
    }


def analyze_calibration(document: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if document.get("status") != "COMPLETED" or not document.get("records"):
        raise ValueError("calibration artifact must be complete and non-empty")
    if document.get("official_test_access") is not False:
        raise ValueError("calibration artifact does not prove official-test isolation")
    records = document["records"]
    epoch_count = min(len(record["history"]) for record in records)
    final_values = [
        float(record["history"][epoch_count - 1]["validation_accuracy"])
        for record in records
    ]
    final_ranks = _ranks(final_values)
    epoch_rows: list[dict[str, Any]] = []
    epoch_summaries = []
    for epoch in range(1, epoch_count + 1):
        values = [float(record["history"][epoch - 1]["validation_accuracy"]) for record in records]
        losses = [float(record["history"][epoch - 1]["validation_loss"]) for record in records]
        ranks = _ranks(values)
        correlation = _spearman(values, final_values)
        epoch_summaries.append(
            {
                "epoch": epoch,
                "spearman_rank_correlation_with_final": correlation,
                "final_top_half_retained_in_current_top_half": len(
                    set(sorted(range(len(records)), key=values.__getitem__, reverse=True)[:3])
                    & set(
                        sorted(range(len(records)), key=final_values.__getitem__, reverse=True)[:3]
                    )
                ),
            }
        )
        for index, record in enumerate(records):
            epoch_rows.append(
                {
                    "architecture_id": record["architecture_id"],
                    "parameter_count": record["parameter_count"],
                    "epoch": epoch,
                    "validation_accuracy": values[index],
                    "validation_loss": losses[index],
                    "rank": ranks[index],
                    "final_rank": final_ranks[index],
                }
            )
    architecture_rows = []
    for index, record in enumerate(records):
        traits = _architecture_traits(record["architecture"])
        architecture_rows.append(
            {
                "architecture_id": record["architecture_id"],
                **traits,
                "parameter_count": record["parameter_count"],
                "duration_seconds": record["duration_seconds"],
                "epoch_2_rank": epoch_rows[(2 - 1) * len(records) + index]["rank"],
                "epoch_4_rank": epoch_rows[(4 - 1) * len(records) + index]["rank"],
                "epoch_6_rank": epoch_rows[(6 - 1) * len(records) + index]["rank"],
                "final_validation_accuracy": final_values[index],
                "final_rank": final_ranks[index],
            }
        )
    return (
        {
            "disclaimer": DISCLAIMER,
            "run_id": document["run_id"],
            "official_test_access": False,
            "architecture_count": len(records),
            "epoch_count": epoch_count,
            "total_duration_seconds": sum(float(record["duration_seconds"]) for record in records),
            "epoch_rank_stability": epoch_summaries,
            "hypothetical_successive_halving": [
                _successive_halving_risk(records, grace, epoch_count) for grace in range(2, 7)
            ],
            "architectures": architecture_rows,
        },
        epoch_rows,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _polyline(points: Iterable[tuple[float, float]], color: str) -> str:
    coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>'


def _calibration_svg(epoch_rows: list[dict[str, Any]], output: Path, *, ranks: bool) -> None:
    width, height = 920, 540
    left, top, plot_width, plot_height = 72, 72, 800, 390
    ids = list(dict.fromkeys(row["architecture_id"] for row in epoch_rows))
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2"]
    epochs = max(int(row["epoch"]) for row in epoch_rows)
    if ranks:
        y_min, y_max, field = 1.0, float(len(ids)), "rank"
        title = "Calibration rank stability by epoch"
    else:
        values = [float(row["validation_accuracy"]) for row in epoch_rows]
        y_min, y_max, field = min(values), max(values), "validation_accuracy"
        if y_min == y_max:
            y_max += 1
        title = "Calibration validation accuracy by epoch"
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" '
        f'font-size="18" font-weight="bold">{title}</text>',
        f'<text x="{width / 2}" y="51" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13" fill="#b91c1c">{DISCLAIMER}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>',
    ]
    for series_index, architecture_id in enumerate(ids):
        rows = [row for row in epoch_rows if row["architecture_id"] == architecture_id]
        points = []
        for row in rows:
            x = left + (int(row["epoch"]) - 1) / max(1, epochs - 1) * plot_width
            normalized = (float(row[field]) - y_min) / (y_max - y_min)
            y = top + (normalized if ranks else 1 - normalized) * plot_height
            points.append((x, y))
        elements.append(_polyline(points, colors[series_index % len(colors)]))
        legend_y = 482 + (series_index // 3) * 20
        legend_x = 80 + (series_index % 3) * 275
        elements.append(
            f'<text x="{legend_x}" y="{legend_y}" font-family="monospace" font-size="12" '
            f'fill="{colors[series_index % len(colors)]}">{architecture_id}</text>'
        )
    elements.append("</svg>")
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _best_so_far(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = float("-inf")
    elapsed = 0.0
    rows = []
    for index, trial in enumerate(trials, start=1):
        elapsed += float(trial["duration_seconds"])
        score = trial.get("validation_accuracy")
        if score is not None:
            best = max(best, float(score))
        rows.append(
            {
                "strategy": trial["search_strategy"],
                "trial_number": index,
                "trial_id": trial["trial_id"],
                "elapsed_trial_seconds_cumulative": elapsed,
                "best_validation_accuracy": None if best == float("-inf") else best,
            }
        )
    return rows


def _trial_traits(trial: dict[str, Any]) -> dict[str, Any]:
    return {
        **_architecture_traits(trial["architecture"]),
        "parameter_count": int(trial["parameter_count"]),
    }


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _late_epoch_summary(document: dict[str, Any]) -> dict[str, Any]:
    histories: dict[str, list[dict[str, Any]]] = {}
    for row in document.get("history", []):
        histories.setdefault(str(row["trial_id"]), []).append(row)
    gains_from_8 = []
    gains_from_10 = []
    trial_rows = []
    for trial in document["trials"]:
        if trial["status"] != "COMPLETED":
            continue
        by_epoch = {int(row["epoch"]): row for row in histories.get(trial["trial_id"], [])}
        final_epoch = int(trial["epochs_consumed"])
        if final_epoch not in by_epoch:
            continue
        final_score = float(by_epoch[final_epoch]["validation_accuracy"])
        gain_8 = final_score - float(by_epoch[8]["validation_accuracy"]) if 8 in by_epoch else None
        gain_10 = (
            final_score - float(by_epoch[10]["validation_accuracy"])
            if 10 in by_epoch
            else None
        )
        if gain_8 is not None:
            gains_from_8.append(gain_8)
        if gain_10 is not None:
            gains_from_10.append(gain_10)
        trial_rows.append(
            {
                "trial_id": trial["trial_id"],
                "final_epoch": final_epoch,
                "gain_epoch_8_to_final": gain_8,
                "gain_epoch_10_to_final": gain_10,
            }
        )
    return {
        "completed_trials_with_history": len(trial_rows),
        "median_gain_epoch_8_to_final": _median(gains_from_8),
        "median_gain_epoch_10_to_final": _median(gains_from_10),
        "positive_gain_epoch_8_to_final_count": sum(gain > 0 for gain in gains_from_8),
        "positive_gain_epoch_10_to_final_count": sum(gain > 0 for gain in gains_from_10),
        "trials": trial_rows,
    }


def _phase_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["validation_accuracy"] is not None]
    best = max(scored, key=lambda row: float(row["validation_accuracy"])) if scored else None
    statuses = {
        status: sum(row["status"] == status for row in rows)
        for status in ("COMPLETED", "PRUNED", "FAILED")
    }
    pruning_iterations: dict[str, int] = {}
    for row in rows:
        if row["pruning_iteration"] is not None:
            key = str(row["pruning_iteration"])
            pruning_iterations[key] = pruning_iterations.get(key, 0) + 1
    return {
        "trial_count": len(rows),
        "completed_count": statuses["COMPLETED"],
        "pruned_count": statuses["PRUNED"],
        "failed_count": statuses["FAILED"],
        "epochs_consumed": sum(int(row["epochs_consumed"]) for row in rows),
        "duration_seconds_sum": sum(float(row["duration_seconds"]) for row in rows),
        "median_parameter_count": _median([float(row["parameter_count"]) for row in rows]),
        "median_validation_accuracy": _median(
            [float(row["validation_accuracy"]) for row in scored]
        ),
        "best_trial_id": best["trial_id"] if best else None,
        "best_validation_accuracy": best["validation_accuracy"] if best else None,
        "best_parameter_count": best["parameter_count"] if best else None,
        "pruning_iterations": pruning_iterations,
    }


def _capacity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["validation_accuracy"] is not None]
    completed = [row for row in scored if row["status"] == "COMPLETED"]

    def correlation(selected: list[dict[str, Any]]) -> float | None:
        return _spearman(
            [float(row["parameter_count"]) for row in selected],
            [float(row["validation_accuracy"]) for row in selected],
        )

    return {
        "terminal_score_parameter_spearman": correlation(scored),
        "completed_score_parameter_spearman": correlation(completed),
        "median_parameter_count": _median([float(row["parameter_count"]) for row in rows]),
        "completed_median_parameter_count": _median(
            [float(row["parameter_count"]) for row in completed]
        ),
    }


def _curated_trials(document: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "architecture": trial["architecture"],
        }
        for row, trial in zip(rows, document["trials"], strict=True)
    ]


def _search_epoch_rows(
    document: dict[str, Any], trial_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    trial_metadata = {row["trial_id"]: row for row in trial_rows}
    rows = []
    for history in document.get("history", []):
        metadata = trial_metadata[history["trial_id"]]
        rows.append(
            {
                "strategy": metadata["strategy"],
                "trial_number": metadata["trial_number"],
                "proposal_phase": metadata["proposal_phase"],
                "trial_id": history["trial_id"],
                "epoch": history["epoch"],
                "parameter_count": history["parameter_count"],
                "validation_accuracy": history["validation_accuracy"],
                "validation_loss": history["validation_loss"],
                "train_loss": history["train_loss"],
                "elapsed_trial_seconds": history["elapsed_trial_seconds"],
            }
        )
    return rows


def _paired_provenance(
    random_manifest: dict[str, Any], tpe_manifest: dict[str, Any]
) -> dict[str, Any]:
    for name, manifest in (("random", random_manifest), ("tpe", tpe_manifest)):
        if manifest.get("status") != "COMPLETED":
            raise ValueError(f"{name} manifest must be complete")
        if manifest.get("git", {}).get("dirty") is not False:
            raise ValueError(f"{name} manifest must record a clean worktree")
    if random_manifest["git"]["commit_sha"] != tpe_manifest["git"]["commit_sha"]:
        raise ValueError("paired pilot manifests must use the same Git commit")
    random_config = copy.deepcopy(random_manifest["config"])
    tpe_config = copy.deepcopy(tpe_manifest["config"])
    random_config["search"]["strategy"] = "<strategy>"
    tpe_config["search"]["strategy"] = "<strategy>"
    random_config["output"]["run_label"] = "<run-label>"
    tpe_config["output"]["run_label"] = "<run-label>"
    if random_config != tpe_config:
        raise ValueError("paired pilot configs differ beyond strategy and run label")
    return {
        "paired_config_match": True,
        "git": random_manifest["git"],
        "software": random_manifest["software"],
        "system": random_manifest["system"],
        "execution_common": {
            key: value
            for key, value in random_manifest["execution"].items()
            if key != "search_strategy"
        },
        "random": {
            "run_id": random_manifest["run_id"],
            "started_at_utc": random_manifest["started_at_utc"],
            "finished_at_utc": random_manifest["finished_at_utc"],
            "status": random_manifest["status"],
        },
        "tpe": {
            "run_id": tpe_manifest["run_id"],
            "started_at_utc": tpe_manifest["started_at_utc"],
            "finished_at_utc": tpe_manifest["finished_at_utc"],
            "status": tpe_manifest["status"],
        },
    }


def analyze_search(document: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trials = document["trials"]
    if not trials:
        raise ValueError("search artifact has no trials")
    durations = [float(trial["duration_seconds"]) for trial in trials]
    completed_durations = [
        float(trial["duration_seconds"]) for trial in trials if trial["status"] == "COMPLETED"
    ]
    best_trial = max(
        (trial for trial in trials if trial.get("validation_accuracy") is not None),
        key=lambda trial: float(trial["validation_accuracy"]),
    )
    trial_rows = []
    for index, trial in enumerate(trials, start=1):
        trial_rows.append(
            {
                "strategy": trial["search_strategy"],
                "trial_number": index,
                "trial_id": trial["trial_id"],
                "proposal_phase": (
                    "startup"
                    if trial["search_strategy"] == "bayesian" and index <= 10
                    else "adaptive"
                    if trial["search_strategy"] == "bayesian"
                    else "random"
                ),
                "status": trial["status"],
                "epochs_consumed": trial["epochs_consumed"],
                "pruning_iteration": trial["pruning_iteration"],
                "validation_accuracy": trial["validation_accuracy"],
                "validation_loss": trial["validation_loss"],
                "train_loss": trial["train_loss"],
                "duration_seconds": trial["duration_seconds"],
                **_trial_traits(trial),
            }
        )
    return (
        {
            "run_id": document["run_id"],
            "strategy": trials[0]["search_strategy"],
            "trial_count": len(trials),
            **document["aggregate"],
            "median_trial_duration_seconds": statistics.median(durations),
            "median_completed_trial_duration_seconds": (
                statistics.median(completed_durations) if completed_durations else None
            ),
            "best_trial_id": best_trial["trial_id"],
            "best_validation_accuracy": best_trial["validation_accuracy"],
            "best_parameter_count": best_trial["parameter_count"],
            "best_so_far": _best_so_far(trials),
            "late_epoch_learning": _late_epoch_summary(document),
            "capacity": _capacity_summary(trial_rows),
            "pruning_iterations": _phase_summary(trial_rows)["pruning_iterations"],
        },
        trial_rows,
    )


def _search_lines_svg(
    rows: list[dict[str, Any]], output: Path, *, elapsed: bool = False
) -> None:
    width, height = 920, 500
    left, top, plot_width, plot_height = 72, 72, 800, 350
    x_field = "elapsed_trial_seconds_cumulative" if elapsed else "trial_number"
    title = (
        "Pilot best-so-far validation accuracy vs cumulative trial time"
        if elapsed
        else "Pilot best-so-far validation accuracy vs trial"
    )
    values = [float(row["best_validation_accuracy"]) for row in rows]
    x_values = [float(row[x_field]) for row in rows]
    y_min, y_max = min(values), max(values)
    if y_min == y_max:
        y_max += 1
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_max += 1
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" '
        f'font-size="18" font-weight="bold">{title}</text>',
        f'<text x="{width / 2}" y="51" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13" fill="#b91c1c">{DISCLAIMER}</text>',
    ]
    for strategy, color in (("random", "#2563eb"), ("bayesian", "#dc2626")):
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        points = [
            (
                left + (float(row[x_field]) - x_min) / (x_max - x_min) * plot_width,
                top
                + (1 - (float(row["best_validation_accuracy"]) - y_min) / (y_max - y_min))
                * plot_height,
            )
            for row in strategy_rows
        ]
        elements.append(_polyline(points, color))
        elements.append(
            f'<text x="{left + (0 if strategy == "random" else 150)}" y="460" '
            f'font-family="sans-serif" font-size="13" fill="{color}">{strategy}</text>'
        )
    elements.append("</svg>")
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _scatter_svg(rows: list[dict[str, Any]], output: Path) -> None:
    rows = [row for row in rows if row["validation_accuracy"] is not None]
    if not rows:
        raise ValueError("parameter scatter requires at least one reported validation score")
    width, height = 920, 500
    left, top, plot_width, plot_height = 72, 72, 800, 350
    x_values = [float(row["parameter_count"]) for row in rows]
    y_values = [float(row["validation_accuracy"]) for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_max = x_max if x_max != x_min else x_max + 1
    y_max = y_max if y_max != y_min else y_max + 1
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" '
        'font-size="18" font-weight="bold">Pilot validation accuracy vs parameter count</text>',
        f'<text x="{width / 2}" y="51" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13" fill="#b91c1c">{DISCLAIMER}</text>',
    ]
    for row in rows:
        x = left + (float(row["parameter_count"]) - x_min) / (x_max - x_min) * plot_width
        y = top + (1 - (float(row["validation_accuracy"]) - y_min) / (y_max - y_min)) * plot_height
        color = "#2563eb" if row["strategy"] == "random" else "#dc2626"
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
    elements.append("</svg>")
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _lifecycle_svg(rows: list[dict[str, Any]], output: Path) -> None:
    width = 920
    height = 90 + len(rows) * 22
    colors = {"COMPLETED": "#059669", "PRUNED": "#ea580c", "FAILED": "#dc2626"}
    max_epochs = max(int(row["epochs_consumed"]) for row in rows)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="26" text-anchor="middle" font-family="sans-serif" '
        'font-size="18" font-weight="bold">Pilot trial lifecycle and pruning</text>',
        f'<text x="{width / 2}" y="49" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13" fill="#b91c1c">{DISCLAIMER}</text>',
    ]
    for index, row in enumerate(rows):
        y = 72 + index * 22
        bar_width = int(row["epochs_consumed"]) / max_epochs * 640
        label = f'{row["strategy"]}:{row["trial_number"]}'
        elements.append(
            f'<text x="10" y="{y + 11}" font-family="monospace" font-size="11">{label}</text>'
        )
        elements.append(
            f'<rect x="170" y="{y}" width="{bar_width:.1f}" height="14" '
            f'fill="{colors[row["status"]]}"/>'
        )
    elements.append("</svg>")
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--random", type=Path)
    parser.add_argument("--tpe", type=Path)
    parser.add_argument("--random-manifest", type=Path)
    parser.add_argument("--tpe-manifest", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    calibration = _read_json(arguments.calibration)
    report, epoch_rows = analyze_calibration(calibration)
    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "calibration_analysis.json", report)
    if arguments.calibration_manifest:
        manifest = _read_json(arguments.calibration_manifest)
        _write_json(
            output / "calibration_provenance.json",
            {
                "run_id": manifest["run_id"],
                "status": manifest["status"],
                "started_at_utc": manifest["started_at_utc"],
                "finished_at_utc": manifest["finished_at_utc"],
                "git": manifest["git"],
                "software": manifest["software"],
                "system": manifest["system"],
                "execution": manifest["execution"],
                "seeds": manifest["seeds"],
                "official_test_access": calibration["official_test_access"],
            },
        )
    _write_csv(output / "calibration_epochs.csv", epoch_rows)
    _write_csv(output / "calibration_architectures.csv", report["architectures"])
    _calibration_svg(epoch_rows, output / "calibration_learning_curves.svg", ranks=False)
    _calibration_svg(epoch_rows, output / "calibration_rank_stability.svg", ranks=True)
    if bool(arguments.random) != bool(arguments.tpe):
        raise ValueError("--random and --tpe must be supplied together")
    if bool(arguments.random_manifest) != bool(arguments.tpe_manifest):
        raise ValueError("--random-manifest and --tpe-manifest must be supplied together")
    if arguments.random and arguments.tpe:
        random_document = _read_json(arguments.random)
        tpe_document = _read_json(arguments.tpe)
        random_report, random_rows = analyze_search(random_document)
        tpe_report, tpe_rows = analyze_search(tpe_document)
        rows = random_rows + tpe_rows
        startup = [
            row
            for row in tpe_rows
            if row["proposal_phase"] == "startup" and row["validation_accuracy"] is not None
        ]
        adaptive = [
            row
            for row in tpe_rows
            if row["proposal_phase"] == "adaptive" and row["validation_accuracy"] is not None
        ]
        report["comparative_pilot"] = {
            "disclaimer": DISCLAIMER,
            "random": random_report,
            "tpe": tpe_report,
            "tpe_behavior": {
                "startup_trials": min(10, len(tpe_rows)),
                "adaptive_trials": max(0, len(tpe_rows) - 10),
                "startup_trials_with_score": len(startup),
                "adaptive_trials_with_score": len(adaptive),
                "best_startup_validation_accuracy": (
                    max(float(row["validation_accuracy"]) for row in startup)
                    if startup
                    else None
                ),
                "best_adaptive_validation_accuracy": (
                    max(float(row["validation_accuracy"]) for row in adaptive)
                    if adaptive
                    else None
                ),
                "startup": _phase_summary(startup),
                "adaptive": _phase_summary(adaptive),
            },
            "capacity": {
                "random": _capacity_summary(random_rows),
                "tpe_startup": _capacity_summary(startup),
                "tpe_adaptive": _capacity_summary(adaptive),
            },
            "paired_startup": {
                "architecture_matches": sum(
                    left["architecture"] == right["architecture"]
                    for left, right in zip(
                        random_document["trials"][:10],
                        tpe_document["trials"][:10],
                        strict=False,
                    )
                ),
                "terminal_score_matches": sum(
                    left["validation_accuracy"] == right["validation_accuracy"]
                    for left, right in zip(random_rows[:10], tpe_rows[:10], strict=False)
                ),
            },
        }
        if arguments.random_manifest and arguments.tpe_manifest:
            provenance = _paired_provenance(
                _read_json(arguments.random_manifest), _read_json(arguments.tpe_manifest)
            )
            report["comparative_pilot"]["provenance"] = provenance
            _write_json(output / "pilot_provenance.json", provenance)
        _write_json(output / "pilot_analysis.json", report)
        _write_json(
            output / "pilot_architectures.json",
            {
                "disclaimer": DISCLAIMER,
                "random_run_id": random_document["run_id"],
                "tpe_run_id": tpe_document["run_id"],
                "trials": _curated_trials(random_document, random_rows)
                + _curated_trials(tpe_document, tpe_rows),
            },
        )
        _write_csv(output / "pilot_trials.csv", rows)
        _write_csv(
            output / "pilot_epochs.csv",
            _search_epoch_rows(random_document, random_rows)
            + _search_epoch_rows(tpe_document, tpe_rows),
        )
        best_rows = random_report["best_so_far"] + tpe_report["best_so_far"]
        _write_csv(output / "pilot_best_so_far.csv", best_rows)
        _search_lines_svg(best_rows, output / "pilot_best_by_trial.svg")
        _search_lines_svg(best_rows, output / "pilot_best_by_time.svg", elapsed=True)
        _scatter_svg(rows, output / "pilot_parameters_vs_accuracy.svg")
        _lifecycle_svg(rows, output / "pilot_trial_lifecycle.svg")


if __name__ == "__main__":
    main()
