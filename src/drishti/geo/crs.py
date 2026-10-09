"""Coordinate reference system handling.

The CRS is DERIVED from the GPS track by default (UTM zone from the median lon/lat) — never hardcoded
(AGENTS.md §5). Actual coordinate transforms use pyproj when available; the UTM-zone math here is pure
and always available so the CRS can be chosen even before pyproj is installed.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DerivedCrs:
    epsg: int
    zone: int
    hemisphere: str  # "N" | "S"
    derived_from: str  # "gps_median_lonlat" | "user_set"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 lat/lon points."""
    r = 6_371_008.8  # mean Earth radius (m)
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def utm_zone_from_lon(lon: float) -> int:
    """UTM zone number 1..60 for a longitude in degrees."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    # wrap/clamp for the antimeridian edge
    return ((zone - 1) % 60) + 1


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing (lon, lat)."""
    zone = utm_zone_from_lon(lon)
    return (32600 if lat >= 0 else 32700) + zone


def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        raise ValueError("no values to take a median of")
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def derive_crs_from_track(
    lons: Sequence[float], lats: Sequence[float], override_epsg: int | None = None
) -> DerivedCrs:
    """Choose a projected CRS for a flight from its GPS track.

    If ``override_epsg`` is given (user set it in a dataset descriptor), it is honored and recorded as
    user-set. Otherwise the UTM zone is computed from the median lon/lat.
    """
    if override_epsg is not None:
        return DerivedCrs(epsg=override_epsg, zone=-1, hemisphere="?", derived_from="user_set")
    if not lons or not lats or len(lons) != len(lats):
        raise ValueError("need matching, non-empty lon/lat sequences to derive a CRS")
    mlon, mlat = _median(lons), _median(lats)
    if not (-180.0 <= mlon <= 180.0 and -90.0 <= mlat <= 90.0):
        raise ValueError(f"median lon/lat out of range: ({mlon}, {mlat}) — check telemetry units")
    epsg = utm_epsg(mlon, mlat)
    return DerivedCrs(
        epsg=epsg,
        zone=utm_zone_from_lon(mlon),
        hemisphere="N" if mlat >= 0 else "S",
        derived_from="gps_median_lonlat",
    )
