"""Similarity (Sim(3)) alignment via the Umeyama algorithm.

Used by the metric spine to align the visual reconstruction frame to the geographic (ENU/UTM) frame
using GPS, recovering rotation, translation, and metric scale without Ground Control Points. Pure
NumPy — always available and unit-tested.

Umeyama, "Least-squares estimation of transformation parameters between two point patterns", 1991.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Similarity:
    scale: float
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        return (self.scale * (self.R @ pts.T)).T + self.t

    def as_matrix(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.scale * self.R
        M[:3, 3] = self.t
        return M


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Similarity:
    """Estimate the similarity transform mapping ``src`` onto ``dst`` (both N×3), minimizing MSE.

    Returns scale s, rotation R, translation t such that  dst ≈ s * R @ src + t.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/dst must be equal N×3 arrays, got {src.shape} and {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError("need at least 3 correspondences for a stable similarity fit")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_src = (src_c**2).sum() / n
        scale = float((D * np.diag(S)).sum() / var_src) if var_src > 0 else 1.0
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)
    return Similarity(scale=scale, R=R, t=t)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))
