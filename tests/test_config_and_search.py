from __future__ import annotations

from pathlib import Path

import optuna
import pytest

from automl_nas.config import ConfigError, load_config
from automl_nas.search import (
    ConditionalArchitectureSpace,
    build_scheduler,
    trial_resources,
)


class RecordingTrial:
    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.names: list[str] = []

    def suggest_int(self, name: str, low: int, high: int) -> int:
        self.names.append(name)
        assert low <= self.depth <= high
        return self.depth

    def suggest_categorical(self, name: str, choices: tuple) -> object:
        self.names.append(name)
        return choices[0]


def test_valid_config_loads(smoke_config) -> None:
    assert smoke_config.data.dataset == "synthetic"
    assert smoke_config.search.num_trials == 2
    assert smoke_config.resources.gpu_per_trial == 0


def test_invalid_config_fails_early(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("seed: -1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="missing keys"):
        load_config(path)


def test_malformed_yaml_fails_as_config_error(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("seed: [unterminated\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot parse YAML config"):
        load_config(path)


@pytest.mark.parametrize("depth", [2, 4])
def test_conditional_space_samples_active_blocks_only(smoke_config, depth: int) -> None:
    trial = RecordingTrial(depth)
    sampled = ConditionalArchitectureSpace(smoke_config)(trial)  # type: ignore[arg-type]
    assert sampled["architecture"]["num_blocks"] == depth
    assert len(sampled["architecture"]["blocks"]) == depth
    assert not any(name.startswith(f"block_{depth}_") for name in trial.names)


def _sample_with_seed(config, seed: int) -> dict:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    trial = study.ask()
    sampled = ConditionalArchitectureSpace(config)(trial)
    study.tell(trial, 0.0)
    return sampled["architecture"]


def test_seeded_optuna_sampling_is_controlled(smoke_config) -> None:
    assert _sample_with_seed(smoke_config, 77) == _sample_with_seed(smoke_config, 77)


def test_asha_and_resources_match_config(smoke_config) -> None:
    scheduler = build_scheduler(smoke_config)
    assert scheduler._metric == "validation_accuracy"
    assert scheduler._mode == "max"
    assert scheduler._time_attr == "training_iteration"
    assert scheduler._max_t == 1
    assert trial_resources(smoke_config) == {"cpu": 1, "gpu": 0}
