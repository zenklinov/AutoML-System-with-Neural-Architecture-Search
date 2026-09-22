"""Command-line interface for validated NAS runs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from automl_nas.config import ConfigError, load_config
from automl_nas.search import run_search
from automl_nas.workflows import run_calibration, run_confirmation, run_final_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="automl-nas")
    subcommands = parser.add_subparsers(dest="command", required=True)

    search_parser = subcommands.add_parser("search", help="run a validated search config")
    search_parser.add_argument("--config", required=True, help="path to a YAML config")
    search_parser.add_argument(
        "--resume-run",
        type=Path,
        help="resume an interrupted run directory using its persisted Ray state",
    )

    matrix_parser = subcommands.add_parser(
        "search-matrix", help="run every approved candidate-search seed"
    )
    matrix_parser.add_argument("--config", required=True)

    calibration_parser = subcommands.add_parser(
        "calibrate-asha", help="collect full curves for a fixed panel"
    )
    calibration_parser.add_argument("--config", required=True)
    calibration_parser.add_argument("--panel", required=True)

    confirmation_parser = subcommands.add_parser(
        "confirm", help="retrain completed shortlisted candidates"
    )
    confirmation_parser.add_argument("--config", required=True)
    confirmation_parser.add_argument("--results", required=True, nargs="+")
    confirmation_parser.add_argument("--shortlist-size", type=int, default=3)

    final_parser = subcommands.add_parser(
        "final-train", help="train selected and baseline models from scratch"
    )
    final_parser.add_argument("--config", required=True)
    final_parser.add_argument("--confirmation", required=True)

    test_parser = subcommands.add_parser(
        "evaluate-locked-test", help="evaluate already-trained final models once"
    )
    test_parser.add_argument("--config", required=True)
    test_parser.add_argument("--final-models", required=True)
    test_parser.add_argument("--acknowledge-locked-test", action="store_true")

    validation_parser = subcommands.add_parser(
        "validate-config", help="validate a YAML config without running a search"
    )
    validation_parser.add_argument("--config", required=True, help="path to a YAML config")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config)
    except ConfigError as error:
        parser.error(str(error))

    if arguments.command == "validate-config":
        print(f"Valid configuration: {config.source_path}")
        return 0
    if arguments.command == "search-matrix":
        for search_seed in config.protocol.candidate_search_seeds:
            run_search(
                replace(config, search=replace(config.search, seed=search_seed)),
                command=["automl-nas", *list(argv or sys.argv[1:]), f"resolved-seed={search_seed}"],
            )
        return 0
    if arguments.command == "calibrate-asha":
        print(run_calibration(config, Path(arguments.panel).resolve()))
        return 0
    if arguments.command == "confirm":
        print(
            run_confirmation(
                config,
                [Path(path).resolve() for path in arguments.results],
                arguments.shortlist_size,
            )
        )
        return 0
    if arguments.command == "final-train":
        print(run_final_training(config, Path(arguments.confirmation).resolve()))
        return 0
    if arguments.command == "evaluate-locked-test":
        from automl_nas.locked_test import run_locked_test_evaluation

        print(
            run_locked_test_evaluation(
                config, Path(arguments.final_models).resolve(), arguments.acknowledge_locked_test
            )
        )
        return 0
    run_search(
        config,
        command=["automl-nas", *list(argv or sys.argv[1:])],
        resume_run_directory=arguments.resume_run,
    )
    return 0
