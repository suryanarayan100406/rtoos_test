"""EXIF GPS telemetry — real, via exiftool (guarded).

For video with embedded GPS, ``exiftool -ee -j`` extracts the GPS track. ``path`` may be a video file
or a directory of geotagged frames. Requires exiftool on PATH.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .base import Telemetry, TelemetryError, TelemetrySample


def _to_deg(value) -> float:
    """exiftool -n returns decimal degrees already; keep float coercion defensive."""
    return float(value)


def parse_exif(path: str | Path) -> Telemetry:
    if shutil.which("exiftool") is None:
        raise TelemetryError(
            "EXIF parsing needs 'exiftool' on PATH (https://exiftool.org). Install it, or use a "
            "dji_srt/csv/mavlink telemetry source instead."
        )
    path = Path(path)
    # -ee extract embedded, -n numeric, -j JSON
    proc = subprocess.run(
        ["exiftool", "-ee", "-n", "-j", "-GPSLatitude", "-GPSLongitude", "-GPSAltitude",
         "-SampleTime", "-GPSDateTime", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise TelemetryError(f"exiftool failed: {proc.stderr.strip()}")
    try:
        records = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise TelemetryError(f"could not parse exiftool JSON: {exc}") from exc

    tel = Telemetry(source_format="exif", source_path=str(path))
    for i, rec in enumerate(records):
        if "GPSLatitude" not in rec or "GPSLongitude" not in rec:
            continue
        tel.samples.append(
            TelemetrySample(
                t=float(rec.get("SampleTime", i)),
                lat=_to_deg(rec["GPSLatitude"]),
                lon=_to_deg(rec["GPSLongitude"]),
                alt=float(rec.get("GPSAltitude", 0.0)),
            )
        )
    if not tel.samples:
        raise TelemetryError(f"no EXIF GPS tags found in: {path}")
    t0 = tel.samples[0].t
    for s in tel.samples:
        s.t -= t0
    tel.fields_present = {"lat", "lon", "alt"}
    return tel
