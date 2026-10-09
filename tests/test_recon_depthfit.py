"""Pure metric-depth geometry: intrinsics, projection, robust scale/shift fit."""
import numpy as np
import pytest

from drishti.recon.depthfit import fit_scale_shift, intrinsics_matrix, project_points


def test_intrinsics_simple_pinhole():
    K = intrinsics_matrix("SIMPLE_PINHOLE", [800.0, 320.0, 240.0])
    assert K[0, 0] == 800 and K[1, 1] == 800
    assert K[0, 2] == 320 and K[1, 2] == 240
    assert K[2, 2] == 1.0


def test_intrinsics_pinhole_two_focals():
    K = intrinsics_matrix("PINHOLE", [800.0, 810.0, 320.0, 240.0])
    assert K[0, 0] == 800 and K[1, 1] == 810


def test_intrinsics_unknown_model_raises():
    with pytest.raises(ValueError):
        intrinsics_matrix("MADE_UP", [1.0, 2.0, 3.0])


def test_project_points_on_optical_axis():
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    R_wc = np.eye(3)
    center = np.zeros(3)
    pts = np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0]])
    uv, z = project_points(K, R_wc, center, pts)
    assert np.allclose(uv[0], [50.0, 50.0])     # on-axis -> principal point
    assert np.allclose(uv[1], [60.0, 50.0])     # x=1 at depth 10 -> +10px
    assert np.allclose(z, [10.0, 10.0])


def test_project_points_empty():
    K = np.eye(3)
    uv, z = project_points(K, np.eye(3), np.zeros(3), np.zeros((0, 3)))
    assert uv.shape == (0, 2) and z.shape == (0,)


def test_fit_scale_shift_exact_line():
    rel = np.linspace(1.0, 5.0, 20)
    metric = 2.0 * rel + 3.0
    a, b, inliers = fit_scale_shift(rel, metric, iters=3, keep=0.8)
    assert a == pytest.approx(2.0, abs=1e-6)
    assert b == pytest.approx(3.0, abs=1e-6)
    assert inliers.sum() >= 3


def test_fit_scale_shift_rejects_outliers():
    rel = np.linspace(1.0, 5.0, 40)
    metric = 2.0 * rel + 3.0
    metric[:5] += 50.0                          # gross outliers
    a, b, inliers = fit_scale_shift(rel, metric, iters=5, keep=0.7)
    assert a == pytest.approx(2.0, abs=0.2)
    assert b == pytest.approx(3.0, abs=0.5)
    assert not inliers[:5].any()                # outliers trimmed


def test_fit_scale_shift_needs_enough_points():
    with pytest.raises(ValueError):
        fit_scale_shift(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
