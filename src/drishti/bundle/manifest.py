"""The project-bundle manifest: the run's spine (provenance, per-stage records, measured timings).

No field is ever fabricated — absent data is null/omitted, never a placeholder value (AGENTS.md §5).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

BUNDLE_VERSION = "1"

StageStatus = Literal["pending", "running", "done", "failed", "degraded", "skipped"]


class InputRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    role: str  # video | gps | imu | baro | intrinsics | rtk | check_points
    sha256: str | None = None
    bytes: int | None = None
    present: bool = True


class CrsInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epsg: int | None = None
    derived_from: str = "gps_median_lonlat"  # or "user_set"
    vertical: str = "ellipsoidal"


class Confidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: float | None = None  # 0..1, None => not computed
    detail_ref: str | None = None  # bundle-relative path to a per-element confidence artifact


class StageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    status: StageStatus = "pending"
    environment: str | None = None  # "local" | "cloud-t4" | ...
    started_utc: str | None = None
    ended_utc: str | None = None
    wall_seconds: float | None = None
    inputs_hash: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    params_hash: str | None = None
    outputs: dict[str, str] = Field(default_factory=dict)  # logical name -> bundle-relative path
    outputs_hash: str | None = None  # combined hash of output files (feeds downstream freshness)
    confidence: Confidence = Field(default_factory=Confidence)
    metrics: dict[str, Any] = Field(default_factory=dict)  # only MEASURED values
    degraded_reason: str | None = None
    error: str | None = None


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bundle_version: str = BUNDLE_VERSION
    drishti_version: str = "0.0.0"
    run_id: str = ""
    created_utc: str = ""
    dataset_name: str | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    dataset_snapshot: dict[str, Any] = Field(default_factory=dict)
    seeds: dict[str, int] = Field(default_factory=dict)
    inputs: list[InputRef] = Field(default_factory=list)
    crs: CrsInfo | None = None
    stages: list[StageRecord] = Field(default_factory=list)

    # -- convenience accessors -------------------------------------------------
    def stage(self, name: str) -> StageRecord | None:
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def upsert_stage(self, record: StageRecord) -> None:
        for i, s in enumerate(self.stages):
            if s.name == record.name:
                self.stages[i] = record
                return
        self.stages.append(record)
