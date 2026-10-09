"""Umeyama Sim(3) alignment (metric spine core)."""
import numpy as np
import pytest

from drishti.geo.align import Similarity, rmse, umeyama


def _rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], float)


def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(30, 3))
    R, s, t = _rot_z(30.0), 2.5, np.array([10.0, -5.0, 3.0])
    dst = (s * (R @ src.T)).T + t

    sim = umeyama(src, dst, with_scale=True)
    assert sim.scale == pytest.approx(2.5, rel=1e-6)
    assert np.allclose(sim.R, R, atol=1e-6)
    assert np.allclose(sim.t, t, atol=1e-6)
    assert np.allclose(sim.apply(src), dst, atol=1e-6)
    assert rmse(sim.apply(src), dst) < 1e-9


def test_umeyama_rigid_without_scale():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(12, 3))
    R, t = _rot_z(-15.0), np.array([1.0, 2.0, -3.0])
    dst = (R @ src.T).T + t
    sim = umeyama(src, dst, with_scale=False)
    assert sim.scale == 1.0
    assert np.allclose(sim.apply(src), dst, atol=1e-6)


def test_as_matrix_is_consistent():
    sim = Similarity(scale=2.0, R=_rot_z(90.0), t=np.array([1.0, 0.0, 0.0]))
    M = sim.as_matrix()
    assert M.shape == (4, 4)
    assert np.allclose(M[:3, :3], 2.0 * _rot_z(90.0))
    assert np.allclose(M[:3, 3], [1.0, 0.0, 0.0])


def test_umeyama_input_validation():
    with pytest.raises(ValueError):
        umeyama(np.zeros((2, 3)), np.zeros((2, 3)))          # < 3 points
    with pytest.raises(ValueError):
        umeyama(np.zeros((5, 3)), np.zeros((4, 3)))          # shape mismatch


def test_rmse_zero_and_positive():
    a = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert rmse(a, a) == 0.0
    b = a + np.array([0.0, 0.0, 1.0])
    assert rmse(a, b) == pytest.approx(1.0)
