"""Projected-coordinate transforms (WGS84 lon/lat <-> a projected CRS), via pyproj — guarded.

Used to turn the GPS track into metric easting/northing so the visual reconstruction can be aligned to
the ground with a Sim(3) fit. pyproj is an optional dependency; without it this fails loudly rather
than approximating the projection.
"""
from __future__ import annotations

from collections.abc import Sequence

from ..runtime.deps import require


def project_lonlat(lons: Sequence[float], lats: Sequence[float], epsg: int) -> tuple[list[float], list[float]]:
    """Project WGS84 lon/lat (deg) to (easting, northing) metres in EPSG:``epsg``."""
    pyproj = require("pyproj", purpose="CRS projection")
    tf = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xs, ys = tf.transform(list(lons), list(lats))
    # Transformer returns scalars for length-1 input; normalise to lists
    if not isinstance(xs, (list, tuple)):
        xs, ys = [xs], [ys]
    return list(xs), list(ys)


def unproject_to_lonlat(xs: Sequence[float], ys: Sequence[float], epsg: int) -> tuple[list[float], list[float]]:
    """Inverse of :func:`project_lonlat`: projected metres -> WGS84 lon/lat (deg)."""
    pyproj = require("pyproj", purpose="CRS projection")
    tf = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lons, lats = tf.transform(list(xs), list(ys))
    if not isinstance(lons, (list, tuple)):
        lons, lats = [lons], [lats]
    return list(lons), list(lats)
