"""Pure rasterization: bin scattered CRS points into north-up grids (DSM/DTM/ortho).

NumPy-only so it is unit-testable without GDAL/rasterio. The S9 stage ([../stages/s9_geo.py]) wraps
these grids with a CRS + affine transform and writes GeoTIFFs. Grid row 0 is the northernmost row
(standard north-up raster orientation).
"""
from __future__ import annotations

import math

import numpy as np


def grid_shape(xmin: float, ymin: float, xmax: float, ymax: float, res: float) -> tuple[int, int]:
    w = max(1, int(math.ceil((xmax - xmin) / res)))
    h = max(1, int(math.ceil((ymax - ymin) / res)))
    return h, w


def _cells(x: np.ndarray, y: np.ndarray, xmin: float, ymax: float, res: float, h: int, w: int):
    col = np.floor((x - xmin) / res).astype(int)
    row = np.floor((ymax - y) / res).astype(int)
    ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    return row, col, ok


def elevation_grid(x, y, z, res: float, agg: str = "max",
                   bounds: tuple[float, float, float, float] | None = None):
    """Aggregate point elevations into a grid. agg in {max,min,mean}. Empty cells => NaN.

    Returns (grid HxW float32, xmin, ymax).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    if len(x) == 0:
        raise ValueError("no points to rasterize")
    xmin, ymin, xmax, ymax = bounds or (x.min(), y.min(), x.max(), y.max())
    h, w = grid_shape(xmin, ymin, xmax, ymax, res)
    row, col, ok = _cells(x, y, xmin, ymax, res, h, w)
    row, col, z = row[ok], col[ok], z[ok]
    flat = row * w + col

    grid = np.full(h * w, np.nan, dtype=np.float64)
    if agg == "max":
        g = np.full(h * w, -np.inf)
        np.maximum.at(g, flat, z)
        grid = np.where(np.isinf(g), np.nan, g)
    elif agg == "min":
        g = np.full(h * w, np.inf)
        np.minimum.at(g, flat, z)
        grid = np.where(np.isinf(g), np.nan, g)
    elif agg == "mean":
        s = np.zeros(h * w)
        c = np.zeros(h * w)
        np.add.at(s, flat, z)
        np.add.at(c, flat, 1.0)
        grid = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    else:
        raise ValueError(f"unknown agg '{agg}'")
    return grid.reshape(h, w).astype(np.float32), float(xmin), float(ymax)


def ortho_grid(x, y, rgb, res: float, bounds=None):
    """Mean RGB per cell -> (H,W,3) uint8 + (xmin, ymax). rgb in [0,1] or [0,255]."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rgb = np.asarray(rgb, float)
    if rgb.max() <= 1.0 + 1e-6:
        rgb = rgb * 255.0
    xmin, ymin, xmax, ymax = bounds or (x.min(), y.min(), x.max(), y.max())
    h, w = grid_shape(xmin, ymin, xmax, ymax, res)
    row, col, ok = _cells(x, y, xmin, ymax, res, h, w)
    flat = (row[ok] * w + col[ok])
    out = np.zeros((h * w, 3))
    cnt = np.zeros(h * w)
    for k in range(3):
        np.add.at(out[:, k], flat, rgb[ok, k])
    np.add.at(cnt, flat, 1.0)
    nz = cnt > 0
    out[nz] /= cnt[nz, None]
    return np.clip(out.reshape(h, w, 3), 0, 255).astype(np.uint8), float(xmin), float(ymax)


# --------------------------------------------------------------------------- streaming rasterization
# The whole-array functions above hold every point in RAM at once. A 100M+-point aerial dense cloud
# (plus the temporaries gridding allocates) blows past even a 30 GB host, and at 0.05 m over a ~1.2 km
# scene the float64 ortho accumulator alone is ~15 GB. These helpers accumulate the SAME grids one
# streamed point chunk at a time (see io.pointcloud.iter_ply_chunks): peak RAM tracks one chunk + the
# (compact float32/uint32) output grids, never the whole cloud. Results are bit-for-bit identical to
# elevation_grid / ortho_grid because per-cell max/min/sum are associative across chunks.


def new_elevation_grid(h: int, w: int, agg: str) -> np.ndarray:
    """Seed a flat (h*w) float32 reduction grid for streaming: -inf for max, +inf for min."""
    if agg not in ("max", "min"):
        raise ValueError(f"streaming elevation agg '{agg}' must be 'max' or 'min'")
    return np.full(h * w, -np.inf if agg == "max" else np.inf, dtype=np.float32)


def add_elevation(acc: np.ndarray, x, y, z, xmin: float, ymax: float, res: float,
                  h: int, w: int, agg: str) -> None:
    """Fold one point chunk into a reduction grid from :func:`new_elevation_grid` (in place)."""
    row, col, ok = _cells(np.asarray(x, float), np.asarray(y, float), xmin, ymax, res, h, w)
    flat = row[ok] * w + col[ok]
    zz = np.asarray(z, float)[ok]
    if agg == "max":
        np.maximum.at(acc, flat, zz)
    else:
        np.minimum.at(acc, flat, zz)


def finish_elevation(acc: np.ndarray, h: int, w: int) -> np.ndarray:
    """Empty cells (still ±inf) -> NaN; return the HxW float32 grid."""
    grid = np.where(np.isinf(acc), np.nan, acc)
    return grid.reshape(h, w).astype(np.float32)


def new_ortho_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Seed (sum, count) accumulators for a streaming RGB mean: uint32 (h*w,3) sum + uint32 (h*w) count.

    uint32 is exact and compact: a 5 cm cloud puts ~1 point in a 5 cm cell, so per-cell sums stay far
    below 2^32 — half the RAM of a float64 accumulator and no floating-point rounding.
    """
    return np.zeros((h * w, 3), dtype=np.uint32), np.zeros(h * w, dtype=np.uint32)


def add_ortho(sum_flat: np.ndarray, cnt_flat: np.ndarray, x, y, rgb, xmin: float, ymax: float,
              res: float, h: int, w: int) -> None:
    """Fold one point chunk (rgb in 0..255) into ortho (sum, count) accumulators (in place)."""
    row, col, ok = _cells(np.asarray(x, float), np.asarray(y, float), xmin, ymax, res, h, w)
    flat = row[ok] * w + col[ok]
    c = np.asarray(rgb)[ok].astype(np.uint32, copy=False)
    for k in range(3):
        np.add.at(sum_flat[:, k], flat, c[:, k])
    np.add.at(cnt_flat, flat, 1)


def finish_ortho(sum_flat: np.ndarray, cnt_flat: np.ndarray, h: int, w: int) -> np.ndarray:
    """Per-cell mean RGB -> (H,W,3) uint8. Channel-by-channel to keep the transient small; empty cells 0.

    Integer floor(sum/count) equals ``ortho_grid``'s float mean truncated by ``astype(uint8)`` for these
    non-negative values, so the orthomosaic is identical to the whole-array path.
    """
    cnt = np.maximum(cnt_flat, 1)  # avoid /0; empty cells have sum 0 -> 0
    out = np.empty((h * w, 3), dtype=np.uint8)
    for k in range(3):
        out[:, k] = (sum_flat[:, k] // cnt).astype(np.uint8)
    return out.reshape(h, w, 3)


def morphological_dtm(dsm: np.ndarray, window_px: int) -> np.ndarray:
    """Estimate a bare-earth DTM from a DSM by grey-erosion (local min) then dilation (opening).

    A simple, dependency-free stand-in for SMRF ground filtering: removes above-ground structure while
    preserving terrain. Real SMRF (PDAL) is used instead when available.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    k = max(1, int(window_px))
    if k == 1:
        return dsm.copy()
    filled = np.where(np.isnan(dsm), np.nanmax(dsm), dsm)
    pad = k // 2
    padded = np.pad(filled, pad, mode="edge")
    win = sliding_window_view(padded, (k, k))[: dsm.shape[0], : dsm.shape[1]]
    eroded = win.min(axis=(-1, -2))          # local min: strip structures
    padded2 = np.pad(eroded, pad, mode="edge")
    win2 = sliding_window_view(padded2, (k, k))[: dsm.shape[0], : dsm.shape[1]]
    opened = win2.max(axis=(-1, -2))         # dilate back: restore terrain level
    return opened.astype(np.float32)
