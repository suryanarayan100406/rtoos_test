import numpy as np


def test_best_view_projection_selects_visible_rgb_and_reports_coverage():
    from drishti.recon.texturing import best_view_vertex_colors

    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[2, 2] = [10, 20, 200]  # BGR; expected RGB is [200, 20, 10]
    views = [{"path": "frame.jpg", "R": np.eye(3), "center_crs": [0.0, 0.0, 0.0]}]
    vertices = np.array([[0.0, 0.0, 5.0]])
    normals = np.array([[0.0, 0.0, -1.0]])
    K = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])

    colors, stats = best_view_vertex_colors(vertices, normals, views, K, lambda _: image)

    assert np.allclose(colors[0], [200 / 255, 20 / 255, 10 / 255])
    assert stats["n_colored_vertices"] == 1
    assert stats["coverage"] == 1.0


def test_best_view_projection_marks_back_facing_surface_uncolored():
    from drishti.recon.texturing import best_view_vertex_colors

    image = np.full((5, 5, 3), 255, dtype=np.uint8)
    views = [{"path": "frame.jpg", "R": np.eye(3), "center_crs": [0.0, 0.0, 0.0]}]
    vertices = np.array([[0.0, 0.0, 5.0]])
    normals = np.array([[0.0, 0.0, 1.0]])
    K = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])

    colors, stats = best_view_vertex_colors(vertices, normals, views, K, lambda _: image)

    assert np.all(colors == 0)
    assert stats["n_colored_vertices"] == 0
    assert stats["coverage"] == 0.0
