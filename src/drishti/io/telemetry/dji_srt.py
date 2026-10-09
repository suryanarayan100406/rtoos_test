"""DJI SRT telemetry parser.

DJI drones embed a per-frame GPS/attitude track in an .SRT subtitle sidecar. Two common layouts are
handled:

  New (bracketed):   [latitude: 22.5431] [longitude: 113.9425] [rel_alt: 0.0 abs_alt: 54.4]
  Old (GPS(...)):    GPS(113.9425,22.5431,20) BAROMETER:54.4

Both are parsed from the SRT blocks; the block's start timecode gives the video-relative time.
"""
from __future__ import annotations

import re
from pathlib import Path

from .base import Telemetry, TelemetryError, TelemetrySample

_TIMECODE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->")
_LAT = re.compile(r"\[?latitude\s*:\s*([-+]?\d+(?:\.\d+)?)\]?", re.I)
_LON = re.compile(r"\[?longitude\s*:\s*([-+]?\d+(?:\.\d+)?)\]?", re.I)
_ABS_ALT = re.compile(r"abs_alt\s*:\s*([-+]?\d+(?:\.\d+)?)", re.I)
_REL_ALT = re.compile(r"rel_alt\s*:\s*([-+]?\d+(?:\.\d+)?)", re.I)
_GPS = re.compile(r"GPS\s*\(\s*([-+]?\d+\.\d+)\s*,\s*([-+]?\d+\.\d+)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)", re.I)
_BARO = re.compile(r"BAROMETER\s*:\s*([-+]?\d+(?:\.\d+)?)", re.I)
_YAW = re.compile(r"\[?(?:gb_)?yaw\s*:\s*([-+]?\d+(?:\.\d+)?)\]?", re.I)
_PITCH = re.compile(r"\[?(?:gb_)?pitch\s*:\s*([-+]?\d+(?:\.\d+)?)\]?", re.I)
_ROLL = re.compile(r"\[?(?:gb_)?roll\s*:\s*([-+]?\d+(?:\.\d+)?)\]?", re.I)


def _timecode_to_seconds(text: str) -> float | None:
    m = _TIMECODE.search(text)
    if not m:
        return None
    hh, mm, ss, ms = m.groups()
    ms = ms.ljust(3, "0")  # normalize 1-3 digit millis
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def parse_dji_srt(path: str | Path) -> Telemetry:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    # SRT blocks are separated by blank lines
    blocks = re.split(r"\n\s*\n", text.strip())
    tel = Telemetry(source_format="dji_srt", source_path=str(path))

    for block in blocks:
        t = _timecode_to_seconds(block)
        lat = lon = alt = None
        rel_alt = None

        gm = _GPS.search(block)
        if gm:  # old GPS(lon,lat,alt) layout
            lon = float(gm.group(1))
            lat = float(gm.group(2))
            alt = float(gm.group(3))
            bm = _BARO.search(block)
            if bm:
                alt = float(bm.group(1))
        else:  # new bracketed layout
            la, lo = _LAT.search(block), _LON.search(block)
            if la and lo:
                lat, lon = float(la.group(1)), float(lo.group(1))
                am = _ABS_ALT.search(block)
                rm = _REL_ALT.search(block)
                if rm:
                    rel_alt = float(rm.group(1))
                alt = float(am.group(1)) if am else (rel_alt if rel_alt is not None else 0.0)

        if lat is None or lon is None or t is None:
            continue  # not a GPS-bearing block

        def _opt(rx: re.Pattern, b: str = block) -> float | None:
            m = rx.search(b)
            return float(m.group(1)) if m else None

        tel.samples.append(
            TelemetrySample(
                t=t, lat=lat, lon=lon, alt=alt if alt is not None else 0.0, rel_alt=rel_alt,
                yaw=_opt(_YAW), pitch=_opt(_PITCH), roll=_opt(_ROLL),
            )
        )

    if not tel.samples:
        raise TelemetryError(
            f"no GPS samples parsed from DJI SRT: {path}. Is it a DJI subtitle track with GPS?"
        )

    # rebase time to zero at the first sample
    t0 = tel.samples[0].t
    for s in tel.samples:
        s.t -= t0

    tel.fields_present = {"lat", "lon", "alt"}
    if any(s.rel_alt is not None for s in tel.samples):
        tel.fields_present.add("rel_alt")
    for f in ("yaw", "pitch", "roll"):
        if any(getattr(s, f) is not None for s in tel.samples):
            tel.fields_present.add(f)
    return tel
