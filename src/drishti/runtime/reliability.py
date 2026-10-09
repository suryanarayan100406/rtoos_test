"""Graceful-degradation ladder (Phase 6 robustness).

Every heavy stage can run in more than one way: the *preferred* method (best model, cloud GPU) down
to a *last-resort* method that still produces a real, labelled result. This module centralises that
policy so degradation is consistent and **honest** across the pipeline:

  * a stage lists its candidate **tiers**, best first, each with an ``available`` flag it computed
    from the runtime (deps present? CUDA present? optional binary on PATH? input produced?);
  * :func:`choose` returns the highest-quality *available* tier, a human ``degraded_reason`` when a
    better tier was skipped, and a **confidence ceiling** for the chosen tier;
  * the stage reports ``min(measured_confidence, ceiling)`` — a fallback can never *claim* the
    confidence of the method it replaced (AGENTS.md §5: measured-only, degraded_reason when degraded).

The ladder never fabricates a result. It only chooses *among methods that genuinely run here*. If a
stage cannot run at all (its floor tier needs something absent), it must still fail loudly via
``require(...)`` / raise — that is not degradation, it is absence, and :func:`choose` says so by
raising :class:`NoViableTier`.

The L0–L6 scale (reference ceilings, not measurements — these are policy caps on how much confidence a
tier may report, deliberately conservative):

    L0  full quality        preferred model(s) + cloud GPU / RTK               cap 1.00
    L1  full method, local   same method on local OpenVINO / CPU (slower)       cap 0.95
    L2  reduced settings      preferred method, lower res / fewer iterations     cap 0.85
    L3  alternative method    classical instead of neural, CPU SfM, etc.         cap 0.70
    L4  prior-only geometry   telemetry / plane / single-view prior, no MVS      cap 0.50
    L5  near-passthrough      capability disabled but pipeline continues         cap 0.35
    L6  minimal viable        last-resort real output, heavily caveated          cap 0.20

Example (a masking stage)::

    from ..runtime.deps import available
    from ..runtime.reliability import choose, tier

    dec = choose([
        tier(0, "sam2_gpu",   available("sam2") and rt.detected.cuda, "SAM2 refinement on GPU"),
        tier(2, "rtdetr_box", available("torch") and available("transformers"), "RT-DETR boxes → masks"),
        tier(5, "no_masking", True, "masking disabled; dynamic objects are NOT removed"),
    ])
    # ... run the branch named by dec.tier.name, measure a real confidence ...
    return StageResult(
        outputs=...,
        degraded_reason=dec.degraded_reason,
        confidence_summary=dec.blend(measured_confidence),
    )
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

# Reference confidence ceilings per ladder level. A tier may report *at most* this much confidence;
# these are conservative policy caps, not measured accuracy.
LEVEL_CAPS: dict[int, float] = {
    0: 1.00,
    1: 0.95,
    2: 0.85,
    3: 0.70,
    4: 0.50,
    5: 0.35,
    6: 0.20,
}

LEVEL_LABELS: dict[int, str] = {
    0: "full quality",
    1: "full method (local)",
    2: "reduced settings",
    3: "alternative method",
    4: "prior-only geometry",
    5: "near-passthrough",
    6: "minimal viable",
}

MAX_LEVEL = 6


class NoViableTier(RuntimeError):
    """Raised when *no* candidate tier is available — the stage genuinely cannot run here.

    This is absence, not degradation: the caller should surface it as a loud failure (a missing
    dependency / input), never as a low-confidence success.
    """


def _cap_for(level: int, override: float | None) -> float:
    if override is not None:
        if not 0.0 <= override <= 1.0:
            raise ValueError(f"confidence cap must be in [0,1], got {override}")
        return override
    if level not in LEVEL_CAPS:
        raise ValueError(f"unknown ladder level {level} (expected 0..{MAX_LEVEL})")
    return LEVEL_CAPS[level]


@dataclass(frozen=True)
class Tier:
    """One rung of a stage's degradation ladder.

    ``level`` is the L0–L6 quality band (0 = best). ``available`` is computed by the stage from the
    runtime *before* calling :func:`choose`. ``confidence_cap`` defaults to the level's reference cap
    but a stage may lower it further (never raise it above the level cap is not enforced — a stage
    that knows better may set any value in [0,1], but by convention keep it ≤ the level cap).
    """
    level: int
    name: str
    available: bool
    note: str = ""
    confidence_cap: float = -1.0  # sentinel; resolved in __post_init__ from LEVEL_CAPS

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tier name must be non-empty")
        object.__setattr__(
            self,
            "confidence_cap",
            _cap_for(self.level, None if self.confidence_cap < 0 else self.confidence_cap),
        )

    def describe(self) -> str:
        state = "available" if self.available else "unavailable"
        band = LEVEL_LABELS.get(self.level, f"L{self.level}")
        tail = f" — {self.note}" if self.note else ""
        return f"L{self.level} {self.name} ({band}, {state}){tail}"


@dataclass(frozen=True)
class Decision:
    """The chosen tier plus an honest audit trail."""
    tier: Tier
    degraded: bool
    degraded_reason: str | None
    confidence_cap: float
    considered: list[str] = field(default_factory=list)

    def blend(self, measured: float | None) -> float | None:
        """Clamp a measured confidence to this decision's ceiling. ``None`` stays ``None``.

        A degraded tier can never report more confidence than its ceiling allows.
        """
        if measured is None:
            return None
        if not 0.0 <= measured <= 1.0:
            raise ValueError(f"measured confidence must be in [0,1], got {measured}")
        return round(min(measured, self.confidence_cap), 4)


def tier(level: int, name: str, available: bool, note: str = "", cap: float | None = None) -> Tier:
    """Convenience constructor. ``cap`` overrides the level's reference confidence ceiling."""
    return Tier(
        level=level,
        name=name,
        available=bool(available),
        note=note,
        confidence_cap=-1.0 if cap is None else cap,
    )


def choose(tiers: Sequence[Tier]) -> Decision:
    """Pick the best (lowest-level) *available* tier.

    Ties on ``level`` are broken by list order (put the more-preferred first). Raises
    :class:`NoViableTier` if the list is empty or nothing is available.
    """
    if not tiers:
        raise NoViableTier("no candidate tiers were supplied")

    considered = [t.describe() for t in tiers]

    # Stable pick: lowest level among available, first-listed wins a tie.
    best: Tier | None = None
    for t in tiers:
        if not t.available:
            continue
        if best is None or t.level < best.level:
            best = t

    if best is None:
        raise NoViableTier(
            "no degradation tier is available in this environment:\n  "
            + "\n  ".join(considered)
        )

    top_level = min(t.level for t in tiers)
    degraded = best.level > top_level
    reason: str | None = None
    if degraded:
        skipped = [
            f"L{t.level} {t.name}"
            for t in tiers
            if t.level < best.level and not t.available
        ]
        band = LEVEL_LABELS.get(best.level, f"L{best.level}")
        reason = f"fell back to '{best.name}' ({band}, L{best.level})"
        if best.note:
            reason += f": {best.note}"
        if skipped:
            reason += f"; higher tiers unavailable: {', '.join(skipped)}"

    return Decision(
        tier=best,
        degraded=degraded,
        degraded_reason=reason,
        confidence_cap=best.confidence_cap,
        considered=considered,
    )
