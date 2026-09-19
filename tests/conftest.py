from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from automl_nas.config import ExperimentConfig, load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def smoke_config(tmp_path: Path) -> ExperimentConfig:
    config = load_config(REPOSITORY_ROOT / "configs" / "smoke.yaml")
    return replace(
        config,
        output=replace(config.output, root_directory=tmp_path / "artifacts"),
    )


def make_architecture(depth: int) -> dict:
    return {
        "num_blocks": depth,
        "blocks": [
            {
                "kernel_size": 3 if index % 2 == 0 else 5,
                "out_channels": (16, 32, 64, 32)[index],
                "activation": "relu" if index % 2 == 0 else "silu",
                "residual": True,
            }
            for index in range(depth)
        ],
    }
