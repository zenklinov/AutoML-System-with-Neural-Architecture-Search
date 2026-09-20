"""Protocol constants and architecture identity helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from automl_nas.models import CandidateCNN

REFERENCE_BASELINE: dict[str, Any] = {
    "num_blocks": 3,
    "blocks": [
        {"kernel_size": 3, "out_channels": 32, "activation": "relu", "residual": True},
        {"kernel_size": 3, "out_channels": 64, "activation": "relu", "residual": True},
        {"kernel_size": 3, "out_channels": 64, "activation": "relu", "residual": True},
    ],
}


def architecture_id(architecture: dict[str, Any]) -> str:
    canonical = json.dumps(architecture, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def trainable_parameter_count(
    architecture: dict[str, Any], input_channels: int = 3, num_classes: int = 10
) -> int:
    model = CandidateCNN(architecture, input_channels, num_classes)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


REFERENCE_BASELINE_PARAMETER_COUNT = trainable_parameter_count(REFERENCE_BASELINE)
