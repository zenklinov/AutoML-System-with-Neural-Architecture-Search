"""PyTorch model components for discrete NAS candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


class ConfigurableConvBlock(nn.Module):
    """A convolutional block whose operation is fixed by a trial config."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation_name: str,
        use_residual: bool,
    ) -> None:
        super().__init__()
        if kernel_size not in {3, 5}:
            raise ValueError(f"unsupported kernel size: {kernel_size}")
        if activation_name not in {"relu", "silu"}:
            raise ValueError(f"unsupported activation: {activation_name}")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("channel counts must be positive")

        self.use_residual = use_residual
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation_name == "relu" else nn.SiLU(inplace=True)
        self.project = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if use_residual and in_channels != out_channels
            else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        outputs = self.act(self.bn(self.conv(inputs)))
        if self.use_residual:
            if self.project is not None:
                identity = self.project(identity)
            outputs = outputs + identity
        return outputs


class CandidateCNN(nn.Module):
    """A discrete CNN instantiated for one multi-trial NAS candidate."""

    def __init__(
        self,
        architecture: Mapping[str, Any],
        input_channels: int = 3,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        blocks = architecture.get("blocks")
        num_blocks = architecture.get("num_blocks")
        if not isinstance(blocks, Sequence) or isinstance(blocks, str | bytes):
            raise ValueError("architecture.blocks must be a sequence")
        if num_blocks not in {2, 3, 4} or len(blocks) != num_blocks:
            raise ValueError("architecture must define exactly 2, 3, or 4 active blocks")

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        layers: list[nn.Module] = []
        current_channels = 32
        for block in blocks:
            output_channels = int(block["out_channels"])
            layers.extend(
                [
                    ConfigurableConvBlock(
                        in_channels=current_channels,
                        out_channels=output_channels,
                        kernel_size=int(block["kernel_size"]),
                        activation_name=str(block["activation"]),
                        use_residual=bool(block["residual"]),
                    ),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            current_channels = output_channels

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(current_channels, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        outputs = self.features(outputs)
        outputs = self.avgpool(outputs)
        outputs = torch.flatten(outputs, 1)
        return self.classifier(outputs)
