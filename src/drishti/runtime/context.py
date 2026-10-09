"""The Runtime object handed to every stage: detected hardware, config, model registry, logger,
and the placement/backend decisions derived from a stage's ComputeNeed.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import DrishtiConfig
from ..logging import get_logger
from ..models.registry import ModelRegistry
from .compute import ComputeNeed, DetectedCompute, Placement, chosen_backend, detect_compute, place


@dataclass
class Runtime:
    cfg: DrishtiConfig
    detected: DetectedCompute
    registry: ModelRegistry
    logger: object = None

    @classmethod
    def build(cls, cfg: DrishtiConfig) -> Runtime:
        return cls(
            cfg=cfg,
            detected=detect_compute(),
            registry=ModelRegistry(cfg.models.cache_dir, cfg.models.allow_reference_only),
            logger=get_logger("drishti"),
        )

    def place(self, need: ComputeNeed) -> Placement:
        return place(need, self.detected, self.cfg.compute.placement)

    def backend(self, need: ComputeNeed) -> str:
        return chosen_backend(need, self.detected, self.cfg.compute.prefer_openvino)
