"""Stage abstraction: the uniform contract every S0–S10 stage implements, plus a stage registry.

A stage reads its inputs from the bundle, writes its outputs back to the bundle, and returns timing +
confidence + measured metrics. It must fail loudly on missing input and never emit fabricated output
(AGENTS.md §5, §10). The runner ([runner.py]) handles ordering, freshness, placement, and timing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..bundle import Bundle
from ..config import DrishtiConfig
from ..runtime import ComputeNeed, Runtime


@dataclass
class StageResult:
    """What a stage returns. ``wall_seconds`` and ``environment`` are filled by the runner."""
    outputs: dict[str, str] = field(default_factory=dict)  # logical name -> bundle-relative path
    metrics: dict[str, Any] = field(default_factory=dict)  # MEASURED values only
    confidence_summary: float | None = None  # 0..1
    confidence_ref: str | None = None
    degraded_reason: str | None = None  # set when a fallback was used (still a real result)


class Stage(ABC):
    """Base class for all pipeline stages."""

    name: str = "stage"
    requires: tuple[str, ...] = ()
    compute: ComputeNeed = ComputeNeed(cpu_only=True)

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        """The config slice this stage actually uses. Recorded for provenance + freshness.

        Override to return only the relevant sub-config so an unrelated config change does not
        needlessly invalidate this stage.
        """
        return {}

    def extra_input_hashes(self, bundle: Bundle, cfg: DrishtiConfig) -> list[str]:
        """Hashes of non-stage inputs this stage depends on (e.g. S0 depends on raw files)."""
        return []

    @abstractmethod
    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        """Do the work. Read from ``bundle``, write artifacts to ``bundle``, return a StageResult."""
        raise NotImplementedError


# --------------------------------------------------------------------------- registry
_STAGES: dict[str, type[Stage]] = {}


def register_stage(cls: type[Stage]) -> type[Stage]:
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} must set a 'name'")
    if cls.name in _STAGES:
        raise ValueError(f"duplicate stage name: {cls.name}")
    _STAGES[cls.name] = cls
    return cls


def get_stage(name: str) -> Stage:
    if name not in _STAGES:
        raise KeyError(f"unknown stage: {name} (known: {sorted(_STAGES)})")
    return _STAGES[name]()


def all_stage_names() -> list[str]:
    return list(_STAGES)


def registered_stages() -> dict[str, type[Stage]]:
    return dict(_STAGES)
