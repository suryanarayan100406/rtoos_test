"""MAVLink telemetry parser (.tlog / .bin) — real, via pymavlink (guarded).

Reads position from GLOBAL_POSITION_INT (preferred) or GPS_RAW_INT. Requires the `telem` extra.
"""
from __future__ import annotations

from pathlib import Path

from ...runtime.deps import require
from .base import Telemetry, TelemetryError, TelemetrySample


def parse_mavlink(path: str | Path) -> Telemetry:
    mavutil = require("pymavlink.mavutil", purpose="MAVLink telemetry parsing")
    tel = Telemetry(source_format="mavlink", source_path=str(path))
    mlog = mavutil.mavlink_connection(str(path))

    while True:
        msg = mlog.recv_match(type=["GLOBAL_POSITION_INT", "GPS_RAW_INT", "ATTITUDE"], blocking=False)
        if msg is None:
            break
        mtype = msg.get_type()
        t = getattr(msg, "_timestamp", None)
        if mtype == "GLOBAL_POSITION_INT":
            tel.samples.append(
                TelemetrySample(
                    t=float(t) if t else 0.0,
                    lat=msg.lat / 1e7,
                    lon=msg.lon / 1e7,
                    alt=msg.alt / 1000.0,  # mm -> m (AMSL)
                    rel_alt=msg.relative_alt / 1000.0,
                    yaw=msg.hdg / 100.0 if getattr(msg, "hdg", 65535) != 65535 else None,
                )
            )
        elif mtype == "GPS_RAW_INT" and not tel.samples:
            tel.samples.append(
                TelemetrySample(
                    t=float(t) if t else 0.0,
                    lat=msg.lat / 1e7, lon=msg.lon / 1e7, alt=msg.alt / 1000.0,
                )
            )

    if not tel.samples:
        raise TelemetryError(f"no position messages found in MAVLink log: {path}")

    tel.samples.sort(key=lambda s: s.t)
    t0 = tel.samples[0].t
    for s in tel.samples:
        s.t -= t0
    tel.fields_present = {"lat", "lon", "alt"}
    if any(s.rel_alt is not None for s in tel.samples):
        tel.fields_present.add("rel_alt")
    return tel
