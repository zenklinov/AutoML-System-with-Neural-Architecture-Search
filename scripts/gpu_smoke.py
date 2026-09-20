"""Verify the local CUDA experiment path, including Ray GPU assignment."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

import ray
import torch
from torch import nn, optim

from automl_nas.models import CandidateCNN
from automl_nas.protocol import REFERENCE_BASELINE


def _one_training_step() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")
    model = CandidateCNN(REFERENCE_BASELINE, 3, 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    inputs = torch.randn(4, 3, 32, 32, device=device)
    targets = torch.tensor([0, 1, 2, 3], device=device)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return {
        "cuda_available": True,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_count_visible": torch.cuda.device_count(),
        "forward_shape": list(outputs.shape),
        "loss": float(loss.detach().cpu()),
        "allocated_memory_bytes": torch.cuda.memory_allocated(),
    }


@ray.remote(num_cpus=1, num_gpus=1)
def _ray_gpu_step() -> dict[str, Any]:
    result = _one_training_step()
    result["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    result["ray_gpu_ids"] = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
    return result


def main() -> None:
    direct = _one_training_step()
    ray_temp = Path.cwd() / "artifacts" / "gpu-smoke-ray"
    ray.init(num_cpus=1, num_gpus=1, include_dashboard=False, _temp_dir=str(ray_temp.resolve()))
    try:
        assigned = ray.get(_ray_gpu_step.remote())
    finally:
        ray.shutdown()
    if len(assigned["ray_gpu_ids"]) != 1 or assigned["device_count_visible"] != 1:
        raise RuntimeError("Ray did not isolate exactly one GPU for the smoke trial")
    print(
        json.dumps(
            {
                "status": "PASSED",
                "python": platform.python_version(),
                "torch": torch.__version__,
                "direct": direct,
                "ray_trial": assigned,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
