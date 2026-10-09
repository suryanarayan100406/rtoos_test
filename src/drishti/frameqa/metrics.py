"""Pure frame-quality math: sharpness, exposure, keyframe selection.

These functions take plain arrays / numbers and have no I/O or heavy deps beyond NumPy, so they are
directly unit-testable without a video decoder. The S1 stage ([../stages/s1_frameqa.py]) wires them to
real decoded frames.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    """BGR uint8 (H,W,3) or already-gray (H,W) -> float64 luma in [0,255]."""
    a = np.asarray(img)
    if a.ndim == 2:
        return a.astype(np.float64)
    if a.ndim == 3 and a.shape[2] >= 3:
        b, g, r = a[..., 0], a[..., 1], a[..., 2]
        return 0.114 * b + 0.587 * g + 0.299 * r
    raise ValueError(f"unexpected image shape {a.shape}")


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Focus measure: variance of the 4-neighbour Laplacian. Higher = sharper.

    Computed with a NumPy stencil (no OpenCV needed) so it is exact and testable.
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] < 3 or g.shape[1] < 3:
        raise ValueError("variance_of_laplacian needs a 2-D image of at least 3x3")
    lap = (
        -4.0 * g[1:-1, 1:-1]
        + g[:-2, 1:-1] + g[2:, 1:-1]
        + g[1:-1, :-2] + g[1:-1, 2:]
    )
    return float(lap.var())


def exposure_clip_fractions(gray: np.ndarray, low: float = 5.0, high: float = 250.0) -> tuple[float, float]:
    """Fraction of pixels clipped black (<= low) and white (>= high)."""
    g = np.asarray(gray, dtype=np.float64)
    n = g.size or 1
    return float((g <= low).sum()) / n, float((g >= high).sum()) / n


def is_accepted(varlap: float, clip_low: float, clip_high: float,
                blur_min: float, exp_low_max: float, exp_high_max: float) -> bool:
    """A frame passes QA if sharp enough and not over/under-exposed."""
    return varlap >= blur_min and clip_low <= exp_low_max and clip_high <= exp_high_max


def select_fixed_stride(candidate_indices: Sequence[int], stride: int,
                        max_keyframes: int = 0) -> list[int]:
    """Every ``stride``-th accepted frame (stride>=1)."""
    stride = max(1, int(stride))
    picked = [candidate_indices[i] for i in range(0, len(candidate_indices), stride)]
    return picked[:max_keyframes] if max_keyframes else picked


def select_by_displacement(
    candidate_indices: Sequence[int],
    displacements: Sequence[float],
    min_disp: float,
    max_keyframes: int = 0,
) -> list[int]:
    """Greedy keyframe pick: accumulate pairwise displacement and emit a keyframe when the running
    parallax since the last keyframe reaches ``min_disp``.

    ``displacements[k]`` is the displacement between candidate k-1 and k (displacements[0] ignored).
    Pure logic — the actual pixel displacements are measured by the stage (ORB) and passed in.
    """
    if not candidate_indices:
        return []
    picked = [candidate_indices[0]]
    accum = 0.0
    for k in range(1, len(candidate_indices)):
        accum += max(0.0, float(displacements[k]))
        if accum >= min_disp:
            picked.append(candidate_indices[k])
            accum = 0.0
            if max_keyframes and len(picked) >= max_keyframes:
                break
    return picked
