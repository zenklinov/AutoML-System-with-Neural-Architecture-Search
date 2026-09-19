"""Command-line interface for validated NAS runs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from automl_nas.config import ConfigError, load_config
from automl_nas.search import run_search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="automl-nas")
    subcommands = parser.add_subparsers(dest="command", required=True)

    search_parser = subcommands.add_parser("search", help="run a validated search config")
    search_parser.add_argument("--config", required=True, help="path to a YAML config")

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
    run_search(config, command=["automl-nas", *list(argv or sys.argv[1:])])
    return 0
