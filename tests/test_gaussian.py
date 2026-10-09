import numpy as np


def test_prepare_metric_view_uses_recentered_world_coordinates():
    from drishti.recon.gaussian import prepare_metric_view

    color = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.full((2, 2), 2.0, dtype=np.float32)
    extr = np.eye(4, dtype=np.float64)
    extr[:3, 3] = [-437021.0, -4572795.0, -248.0]
    K = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    view, points, _ = prepare_metric_view(
        color, depth, extr, K, 2, 2, np.array([437021.0, 4572795.0, 248.0]), 100,
        np.random.default_rng(0),
    )
    assert np.allclose(view["extr"][:3, 3], [0.0, 0.0, 0.0])
    assert np.all(np.abs(points) < 10.0)
