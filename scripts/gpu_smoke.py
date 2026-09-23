"""Verify deterministic local CUDA training and one isolated Ray GPU worker."""

from __future__ import annotations

import json
import os
import platform
import shutil
import tempfile
import warnings
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import ray
import torch
from torch import nn, optim

from automl_nas.models import CandidateCNN
from automl_nas.protocol import REFERENCE_BASELINE


def _one_training_step() -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG was not set before CUDA training")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.manual_seed(4242)
        torch.cuda.manual_seed_all(4242)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
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
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }


@ray.remote(num_cpus=1, num_gpus=1)
def _ray_gpu_step() -> dict[str, Any]:
    import os as worker_os

    import ray as worker_ray

    from scripts.gpu_smoke import _one_training_step as worker_training_step

    result = worker_training_step()
    result["cuda_visible_devices"] = worker_os.environ.get("CUDA_VISIBLE_DEVICES")
    result["ray_gpu_ids"] = worker_ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
    return result


def _validate(result: dict[str, Any], *, ray_worker: bool) -> None:
    if result["cuda_runtime"] != "12.4":
        raise RuntimeError(f"unexpected CUDA runtime: {result['cuda_runtime']}")
    if "RTX 4060" not in result["device_name"]:
        raise RuntimeError(f"unexpected GPU: {result['device_name']}")
    if result["forward_shape"] != [4, 10] or not result["deterministic_algorithms"]:
        raise RuntimeError("CUDA training or deterministic mode validation failed")
    if ray_worker and (len(result["ray_gpu_ids"]) != 1 or result["device_count_visible"] != 1):
        raise RuntimeError("Ray did not isolate exactly one GPU for the smoke worker")


def main() -> None:
    direct = _one_training_step()
    _validate(direct, ray_worker=False)
    ray_temp = tempfile.mkdtemp(prefix="automl-nas-ray-", dir="/tmp")
    ray.init(num_cpus=1, num_gpus=1, include_dashboard=False, _temp_dir=ray_temp)
    try:
        cluster_resources = ray.cluster_resources()
        assigned = ray.get(_ray_gpu_step.remote())
    finally:
        ray.shutdown()
        shutil.rmtree(ray_temp, ignore_errors=True)
    _validate(assigned, ray_worker=True)
    if cluster_resources.get("GPU") != 1.0:
        raise RuntimeError(f"Ray GPU resource count is not one: {cluster_resources}")
    print(
        json.dumps(
            {
                "status": "PASSED",
                "python": platform.python_version(),
                "torch": torch.__version__,
                "ray": ray.__version__,
                "direct": direct,
                "ray_cluster_gpu_resources": cluster_resources.get("GPU"),
                "ray_worker": assigned,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
