"""Compute detection and the local-vs-cloud placement policy.

The binding constraint: no local discrete GPU (Intel Ultra 7 155H — CPU + Arc iGPU/NPU via OpenVINO)
plus a free cloud T4 (~16 GB) for heavy neural stages. A stage declares a ComputeNeed; the runner
places it per this policy + detected hardware (docs/implementation/02-SYSTEM-DESIGN.md §6).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from .deps import available

Placement = Literal["local", "cloud-t4"]


@dataclass(frozen=True)
class ComputeNeed:
    """What a stage needs to run. The runner uses this to place it."""
    vram_gb: float = 0.0
    needs_cuda: bool = False
    openvino_ok: bool = False
    cpu_only: bool = False


@dataclass
class DetectedCompute:
    cpu_count: int
    ram_gb: float
    cuda: bool
    cuda_name: str | None
    cuda_vram_gb: float
    openvino: bool
    openvino_devices: list[str] = field(default_factory=list)
    is_cloud_gpu: bool = False  # true when a CUDA GPU is present (Colab/Kaggle T4 session)

    def as_dict(self) -> dict:
        return {
            "cpu_count": self.cpu_count,
            "ram_gb": round(self.ram_gb, 1),
            "cuda": self.cuda,
            "cuda_name": self.cuda_name,
            "cuda_vram_gb": round(self.cuda_vram_gb, 1),
            "openvino": self.openvino,
            "openvino_devices": self.openvino_devices,
            "is_cloud_gpu": self.is_cloud_gpu,
        }


def detect_compute() -> DetectedCompute:
    """Best-effort hardware probe. Never fabricates — reports only what it can confirm."""
    cpu_count = os.cpu_count() or 1
    ram_gb = 0.0
    try:
        import psutil  # core dep

        ram_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass

    cuda = False
    cuda_name: str | None = None
    cuda_vram = 0.0
    if available("torch"):
        try:
            import torch

            if torch.cuda.is_available():
                cuda = True
                cuda_name = torch.cuda.get_device_name(0)
                cuda_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            pass

    ov = available("openvino")
    ov_devices: list[str] = []
    if ov:
        try:
            import openvino as ovlib

            ov_devices = list(ovlib.Core().available_devices)
        except Exception:
            ov_devices = []

    return DetectedCompute(
        cpu_count=cpu_count,
        ram_gb=ram_gb,
        cuda=cuda,
        cuda_name=cuda_name,
        cuda_vram_gb=cuda_vram,
        openvino=ov,
        openvino_devices=ov_devices,
        is_cloud_gpu=cuda,
    )


def place(need: ComputeNeed, detected: DetectedCompute, policy: str = "auto") -> Placement:
    """Decide where a stage runs. 'force_local'/'force_cloud' override the auto policy."""
    if policy == "force_local":
        return "local"
    if policy == "force_cloud":
        return "cloud-t4"
    # auto
    if need.cpu_only:
        return "local"
    if need.needs_cuda or need.vram_gb > 0:
        # if this process actually has a CUDA GPU (we're the cloud session), run here
        if detected.cuda and detected.cuda_vram_gb + 0.5 >= need.vram_gb:
            return "cloud-t4"
        # otherwise it belongs on the cloud tier (a separate Colab/Kaggle session)
        return "cloud-t4"
    return "local"


def chosen_backend(need: ComputeNeed, detected: DetectedCompute, prefer_openvino: bool) -> str:
    """Pick an inference backend for a placed stage."""
    if detected.cuda and (need.needs_cuda or need.vram_gb > 0):
        return "torch-cuda"
    if need.openvino_ok and detected.openvino and prefer_openvino:
        return "openvino"
    return "cpu"
