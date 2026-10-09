"""Evidence-driven best-view color projection for mesh vertices.

This is a real multi-view color pass used before the optional UV atlas backend. It projects each mesh
vertex into every registered camera, rejects points outside the image/behind the camera/back-facing the
camera, and samples the highest-scoring observation. It improves the shipped vertex-color fallback while
keeping the artifact label honest: it is not presented as a UV atlas.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .depthfit import project_points


def best_view_vertex_colors(
    vertices_world: np.ndarray,
    normals_world: np.ndarray,
    views: Sequence[dict],
    K: np.ndarray,
    read_image: Callable[[str], np.ndarray],
) -> tuple[np.ndarray, dict[str, int | str | float]]:
    """Return RGB colors selected from real view projections plus measured coverage statistics."""
    vertices = np.asarray(vertices_world, dtype=float)
    normals = np.asarray(normals_world, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices_world must be Nx3, got {vertices.shape}")
    if normals.shape != vertices.shape:
        raise ValueError("normals_world must have the same shape as vertices_world")
    colors = np.zeros((len(vertices), 3), dtype=np.float32)
    scores = np.full(len(vertices), -np.inf, dtype=np.float32)
    assigned = np.zeros(len(vertices), dtype=bool)
    views_used = 0

    for view in views:
        image = read_image(view["path"])
        h, w = image.shape[:2]
        R = np.asarray(view["R"], dtype=float)
        center = np.asarray(view["center_crs"], dtype=float)
        uv, z = project_points(K, R, center, vertices)
        u = np.rint(uv[:, 0]).astype(int)
        v = np.rint(uv[:, 1]).astype(int)
        ray = center.reshape(1, 3) - vertices
        ray_norm = np.linalg.norm(ray, axis=1)
        ray /= np.maximum(ray_norm[:, None], 1e-9)
        normal_norm = np.linalg.norm(normals, axis=1)
        unit_normals = normals / np.maximum(normal_norm[:, None], 1e-9)
        facing = np.einsum("ij,ij->i", unit_normals, ray)
        valid = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (facing > 0)
        if not np.any(valid):
            continue
        views_used += 1
        # Prefer front-facing observations and avoid the outermost pixel border, where reprojection
        # and lens-distortion errors are largest. The score is an evidence ranking, not an accuracy claim.
        border = np.min(np.stack([u, v, w - 1 - u, h - 1 - v], axis=0), axis=0).astype(float)
        score = np.where(valid, facing + 0.001 * np.clip(border, 0.0, None), -np.inf)
        replace = score > scores
        if np.any(replace):
            rgb = image[v[replace], u[replace], ::-1].astype(np.float32) / 255.0
            colors[replace] = rgb
            scores[replace] = score[replace]
            assigned[replace] = True

    return colors, {
        "method": "best_view_vertex_projection",
        "n_vertices": int(len(vertices)),
        "n_colored_vertices": int(assigned.sum()),
        "coverage": round(float(assigned.mean()) if len(assigned) else 0.0, 4),
        "n_views_used": views_used,
    }
