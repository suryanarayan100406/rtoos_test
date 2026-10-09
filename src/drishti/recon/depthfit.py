"""Pure geometry for metric depth alignment: intrinsics, projection, robust scale/shift fit.

Monocular depth (Depth Anything V2) is *relative*. S4 makes it metric by fitting a linear scale+shift
to the sparse metric points from S2 that reproject into each frame. These helpers are NumPy-only and
unit-testable; the S4 stage ([../stages/s4_depth.py]) supplies real network output.
"""
from __future__ import annotations

import numpy as np


def intrinsics_matrix(model: str, params: list[float]) -> np.ndarray:
    """Build a 3x3 pinhole K from a COLMAP camera model + params."""
    p = list(map(float, params))
    m = (model or "").upper()
    if m in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE"):
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    elif m in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        raise ValueError(f"unsupported camera model '{model}' with params {params}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def project_points(K: np.ndarray, R_wc: np.ndarray, center: np.ndarray, pts_world: np.ndarray):
    """Project ground-frame points into a camera.

    ``R_wc`` maps world directions to camera; ``center`` is the camera center in the world frame.
    Returns (uv Nx2 float, z_cam N,) — caller filters z>0 and in-bounds.
    """
    if len(pts_world) == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    cam = (R_wc @ (pts_world - center.reshape(1, 3)).T).T  # N x 3
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uvw = (K @ cam.T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, z


def fit_scale_shift(rel: np.ndarray, metric: np.ndarray, iters: int = 3, keep: float = 0.8):
    """Robustly fit metric ≈ a·rel + b via iterative trimmed least squares.

    Returns (a, b, inlier_mask). Trims the worst-residual points each iteration to resist outliers
    (bad matches, points on dynamic objects).
    """
    rel = np.asarray(rel, dtype=float).ravel()
    metric = np.asarray(metric, dtype=float).ravel()
    mask = np.isfinite(rel) & np.isfinite(metric) & (metric > 0)
    if mask.sum() < 3:
        raise ValueError("need >=3 valid (rel, metric) pairs to fit scale/shift")
    idx = np.where(mask)[0]
    a, b = 1.0, 0.0
    for _ in range(max(1, iters)):
        A = np.column_stack([rel[idx], np.ones(len(idx))])
        (a, b), *_ = np.linalg.lstsq(A, metric[idx], rcond=None)
        resid = np.abs(a * rel[idx] + b - metric[idx])
        if len(idx) <= 3:
            break
        thresh = np.quantile(resid, keep)
        keep_mask = resid <= max(thresh, 1e-9)
        if keep_mask.sum() < 3:
            break
        idx = idx[keep_mask]
    inliers = np.zeros_like(rel, dtype=bool)
    inliers[idx] = True
    return float(a), float(b), inliers


def align_depth(predicted, anchor_uv, anchor_z, *, inverse: bool, min_anchors: int):
    """Anchor either camera-Z depth or relative disparity to metres; never emit unscaled relative units.

    DA V2's relative output behaves as inverse depth. Fitting it linearly to camera-Z bends geometry;
    it must instead be fitted to 1/Z, then inverted. Nonfinite/behind-camera/out-of-image anchors and
    invalid predicted pixels are excluded. Insufficient/degenerate support is an explicit error.
    """
    predicted = np.asarray(predicted, dtype=np.float32)
    uv = np.asarray(anchor_uv, dtype=float)
    z = np.asarray(anchor_z, dtype=float)
    h, w = predicted.shape
    visible = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0)
    uv, z = uv[visible], z[visible]
    u, v = np.rint(uv).astype(np.int64).T if len(uv) else (np.array([], int), np.array([], int))
    inside = (u >= 0) & (v >= 0) & (u < w) & (v < h)
    u, v, z = u[inside], v[inside], z[inside]
    x = predicted[v, u]
    valid = np.isfinite(x) & (x > 0)
    x, z = x[valid], z[valid]
    if len(x) < min_anchors or np.ptp(x) <= np.finfo(np.float32).eps:
        raise ValueError(f"insufficient or degenerate metric anchors ({len(x)})")
    target = 1.0 / z if inverse else z
    a, b, inliers = fit_scale_shift(x, target)
    fitted = a * predicted + b
    valid_pixels = np.isfinite(predicted) & (predicted > 0) & np.isfinite(fitted) & (fitted > 0)
    metric = np.zeros_like(predicted)
    metric[valid_pixels] = 1.0 / fitted[valid_pixels] if inverse else fitted[valid_pixels]
    at_anchors = a * x[inliers] + b
    if np.any(at_anchors <= 0):
        raise ValueError("metric alignment produced nonpositive anchor depths")
    estimated_z = 1.0 / at_anchors if inverse else at_anchors
    residual = float(np.median(np.abs(estimated_z - z[inliers])))
    return metric, {"scale": a, "shift": b, "n_anchors": len(x), "n_inliers": int(inliers.sum()),
                    "resid_m": residual, "inverse_depth_fit": inverse}
