"""Generic CSV telemetry parser with a configurable column mapping (from the dataset descriptor)."""
from __future__ import annotations

import csv as _csv
from datetime import datetime
from pathlib import Path

from ...config import CSVMapping
from .base import Telemetry, TelemetryError, TelemetrySample


def _to_seconds(value: str, first: float | None) -> float:
    """Parse a time cell as float seconds, or ISO-8601 rebased to the first timestamp."""
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TelemetryError(f"unrecognized time value: {value!r}") from exc
    epoch = dt.timestamp()
    return epoch if first is None else epoch


def _get(row: dict[str, str], header: list[str], key: str) -> str | None:
    """Fetch a column by name or 0-based index string."""
    if key in row:
        return row[key]
    if key.isdigit():
        idx = int(key)
        if 0 <= idx < len(header):
            return row[header[idx]]
    return None


def parse_csv(path: str | Path, mapping: CSVMapping | None) -> Telemetry:
    m = mapping or CSVMapping()
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = _csv.DictReader(fh)
        if reader.fieldnames is None:
            raise TelemetryError(f"CSV has no header row: {path}")
        header = list(reader.fieldnames)
        rows = list(reader)
    if not rows:
        raise TelemetryError(f"CSV has no data rows: {path}")

    tel = Telemetry(source_format="csv", source_path=str(path))
    raw_times: list[float] = []

    def _opt(row: dict, name: str | None) -> float | None:
        if not name:
            return None
        v = _get(row, header, name)
        return float(v) if v not in (None, "") else None

    for row in rows:
        lat_s = _get(row, header, m.lat)
        lon_s = _get(row, header, m.lon)
        t_s = _get(row, header, m.time)
        if lat_s is None or lon_s is None or t_s is None:
            raise TelemetryError(
                f"CSV missing required column(s): time={m.time!r} lat={m.lat!r} lon={m.lon!r}. "
                f"Available: {header}"
            )
        alt_s = _get(row, header, m.alt)
        t = _to_seconds(t_s, raw_times[0] if raw_times else None)
        raw_times.append(t)

        tel.samples.append(
            TelemetrySample(
                t=t, lat=float(lat_s), lon=float(lon_s),
                alt=float(alt_s) if alt_s not in (None, "") else 0.0,
                yaw=_opt(row, m.yaw), pitch=_opt(row, m.pitch), roll=_opt(row, m.roll),
            )
        )

    t0 = tel.samples[0].t
    for s in tel.samples:
        s.t -= t0

    tel.fields_present = {"lat", "lon", "alt"}
    for f in ("yaw", "pitch", "roll"):
        if any(getattr(s, f) is not None for s in tel.samples):
            tel.fields_present.add(f)
    return tel
