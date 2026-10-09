"""Optional GTSAM metric-spine optimization.

The optimizer here is intentionally small and explicit: one 3-D camera-center variable per registered
pose, GNSS position factors for frames with telemetry, and visual relative-motion factors between adjacent
poses.  It is a real nonlinear factor graph, not a dependency probe.  Orientation/IMU factors remain a
separate adapter because their file formats and camera-to-IMU extrinsics are dataset-specific; callers
must report that limitation when IMU data exists.
"""
from __future__ import annotations

import numpy as np


def optimize_position_spine(
    initial_local: np.ndarray,
    gnss_local_by_pose: dict[int, np.ndarray],
    *,
    horizontal_sigma_m: float,
    vertical_sigma_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Optimize camera centers with GTSAM and return positions plus measured graph metrics."""
    import gtsam

    initial = np.asarray(initial_local, dtype=float)
    if initial.ndim != 2 or initial.shape[1] != 3 or len(initial) < 2:
        raise ValueError("GTSAM spine needs at least two Nx3 initial positions")
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    def key(index: int):
        return gtsam.symbol("x", int(index))

    # Keep all values close to the origin; this avoids conditioning problems from UTM magnitudes.
    for i, pos in enumerate(initial):
        values.insert(key(i), gtsam.Point3(*map(float, pos)))

    sigmas = np.array([
        max(float(horizontal_sigma_m), 1e-3),
        max(float(horizontal_sigma_m), 1e-3),
        max(float(vertical_sigma_m), 1e-3),
    ])
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    visual_noise = gtsam.noiseModel.Isotropic.Sigma(3, max(float(horizontal_sigma_m), 1e-3))
    graph.add(gtsam.PriorFactorPoint3(key(0), gtsam.Point3(*map(float, initial[0])), prior_noise))
    for i in range(len(initial) - 1):
        delta = initial[i + 1] - initial[i]
        graph.add(gtsam.BetweenFactorPoint3(
            key(i), key(i + 1), gtsam.Point3(*map(float, delta)), visual_noise,
        ))
    for index, gps in gnss_local_by_pose.items():
        if not 0 <= index < len(initial):
            continue
        point = np.asarray(gps, dtype=float).reshape(3)
        graph.add(gtsam.PriorFactorPoint3(key(index), gtsam.Point3(*map(float, point)), prior_noise))

    optimized = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
    result = np.asarray([
        np.asarray(optimized.atPoint3(key(i)), dtype=float).reshape(3)
        for i in range(len(initial))
    ])
    gnss_residuals = [
        float(np.linalg.norm(result[i] - np.asarray(gps, dtype=float)))
        for i, gps in gnss_local_by_pose.items() if 0 <= i < len(result)
    ]
    details: dict[str, object] = {
        "optimizer": "gtsam_levenberg_marquardt",
        "n_variables": int(len(initial)),
        "n_gnss_factors": int(len(gnss_residuals)),
        "n_visual_factors": int(max(0, len(initial) - 1)),
        "postfit_gnss_rmse_m": float(np.sqrt(np.mean(np.square(gnss_residuals))))
        if gnss_residuals else None,
    }
    return result, details
