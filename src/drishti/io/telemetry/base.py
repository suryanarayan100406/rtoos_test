"""Telemetry model + parser dispatch.

Parsers turn real flight logs into a common ``Telemetry`` series. Mandatory fields absent from a log
cause a loud failure upstream (the ingest stage), never a placeholder track.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from pathlib import Path

from ...config import TelemetrySpec


@dataclass
class TelemetrySample:
    t: float  # seconds from the start of the log (video-relative)
    lat: float  # degrees
    lon: float  # degrees
    alt: float  # metres (absolute/ellipsoidal if known)
    rel_alt: float | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None


@dataclass
class Telemetry:
    samples: list[TelemetrySample] = field(default_factory=list)
    source_format: str = ""
    source_path: str = ""
    fields_present: set[str] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        return (self.samples[-1].t - self.samples[0].t) if self.samples else 0.0

    def lons(self) -> list[float]:
        return [s.lon for s in self.samples]

    def lats(self) -> list[float]:
        return [s.lat for s in self.samples]

    def alts(self) -> list[float]:
        return [s.alt for s in self.samples]

    def times(self) -> list[float]:
        return [s.t for s in self.samples]

    def sample_at(self, t: float) -> TelemetrySample:
        """Linear interpolation of the track at video-time ``t`` (clamped to the log range)."""
        if not self.samples:
            raise ValueError("empty telemetry")
        ts = self.times()
        if t <= ts[0]:
            return self.samples[0]
        if t >= ts[-1]:
            return self.samples[-1]
        i = bisect.bisect_left(ts, t)
        a, b = self.samples[i - 1], self.samples[i]
        span = (b.t - a.t) or 1.0
        w = (t - a.t) / span

        def lerp(x: float, y: float) -> float:
            return x + w * (y - x)

        def olerp(x: float | None, y: float | None) -> float | None:
            return lerp(x, y) if (x is not None and y is not None) else (x if x is not None else y)

        return TelemetrySample(
            t=t,
            lat=lerp(a.lat, b.lat),
            lon=lerp(a.lon, b.lon),
            alt=lerp(a.alt, b.alt),
            rel_alt=olerp(a.rel_alt, b.rel_alt),
            yaw=olerp(a.yaw, b.yaw),
            pitch=olerp(a.pitch, b.pitch),
            roll=olerp(a.roll, b.roll),
        )

    def to_jsonable(self) -> dict:
        return {
            "source_format": self.source_format,
            "source_path": self.source_path,
            "fields_present": sorted(self.fields_present),
            "n": len(self.samples),
            "duration_s": round(self.duration, 3),
            "samples": [
                {
                    k: v
                    for k, v in {
                        "t": round(s.t, 3),
                        "lat": s.lat,
                        "lon": s.lon,
                        "alt": s.alt,
                        "rel_alt": s.rel_alt,
                        "yaw": s.yaw,
                        "pitch": s.pitch,
                        "roll": s.roll,
                    }.items()
                    if v is not None
                }
                for s in self.samples
            ],
        }


class TelemetryError(RuntimeError):
    pass


def parse_telemetry(spec: TelemetrySpec) -> Telemetry:
    """Dispatch to the right parser. Raises TelemetryError with a clear message on failure."""
    if spec.path is None:
        raise TelemetryError(f"telemetry format '{spec.format}' requires a 'path'")
    path = Path(spec.path)
    if not path.is_file():
        raise TelemetryError(f"telemetry file not found: {path}")

    if spec.format == "dji_srt":
        from .dji_srt import parse_dji_srt

        return parse_dji_srt(path)
    if spec.format == "csv":
        from .csv_parser import parse_csv

        return parse_csv(path, spec.csv)
    if spec.format == "mavlink":
        from .mavlink import parse_mavlink

        return parse_mavlink(path)
    if spec.format == "exif":
        from .exif import parse_exif

        return parse_exif(path)
    raise TelemetryError(f"unknown telemetry format: {spec.format}")
