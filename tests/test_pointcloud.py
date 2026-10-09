"""Dependency-free PLY/OBJ writers + PLY reader round-trip."""
import numpy as np
import pytest

from drishti.io.pointcloud import (
    StreamingPlyWriter,
    iter_ply_chunks,
    read_ply_points,
    write_obj_mesh,
    write_ply_points,
)


def test_ply_roundtrip_no_color(tmp_path):
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    p = tmp_path / "pc.ply"
    write_ply_points(p, pts)
    r, c = read_ply_points(p)
    assert c is None
    assert np.allclose(r, pts, atol=1e-5)


def test_ply_roundtrip_with_color(tmp_path):
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    cols = np.array([[255, 0, 0], [0, 128, 255]], dtype=float)
    p = tmp_path / "pcc.ply"
    write_ply_points(p, pts, cols)
    r, c = read_ply_points(p)
    assert np.allclose(r, pts, atol=1e-5)
    assert c is not None and c.dtype == np.uint8
    assert list(c[0]) == [255, 0, 0]
    assert list(c[1]) == [0, 128, 255]


def test_ply_color_normalized_0_1_scaled(tmp_path):
    pts = np.array([[0.0, 0.0, 0.0]])
    cols = np.array([[1.0, 0.0, 0.5]])          # 0..1 -> *255
    p = tmp_path / "n.ply"
    write_ply_points(p, pts, cols)
    _, c = read_ply_points(p)
    assert list(c[0]) == [255, 0, 127]


def test_obj_mesh_is_one_indexed(tmp_path):
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    p = tmp_path / "m.obj"
    write_obj_mesh(p, v, faces)
    text = p.read_text(encoding="ascii")
    assert "v 0.000000 0.000000 0.000000" in text
    assert "f 1 2 3" in text


def test_streaming_writer_matches_write_ply_points(tmp_path):
    # Streaming several chunks must produce byte-identical output to writing the concatenation at once,
    # so the RAM-bounded S7 path yields exactly the cloud the in-memory path would have.
    a_pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    a_cols = np.array([[255, 0, 0], [0, 128, 255]], dtype=float)
    b_pts = np.array([[4.0, 5.0, 6.0]])
    b_cols = np.array([[10, 20, 30]], dtype=float)

    streamed = tmp_path / "streamed.ply"
    with StreamingPlyWriter(streamed, with_color=True) as w:
        w.add(a_pts, a_cols)
        w.add(b_pts, b_cols)
    assert w.count == 3

    oneshot = tmp_path / "oneshot.ply"
    write_ply_points(oneshot, np.vstack([a_pts, b_pts]), np.vstack([a_cols, b_cols]))
    assert streamed.read_bytes() == oneshot.read_bytes()

    r, c = read_ply_points(streamed)
    assert np.allclose(r, np.vstack([a_pts, b_pts]), atol=1e-5)
    assert list(c[0]) == [255, 0, 0] and list(c[2]) == [10, 20, 30]


def test_streaming_writer_tracks_count_and_bbox(tmp_path):
    with StreamingPlyWriter(tmp_path / "b.ply", with_color=True) as w:
        w.add(np.array([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0]]), np.array([[1, 1, 1], [2, 2, 2]], dtype=float))
        w.add(np.array([[5.0, 4.0, -6.0]]), np.array([[3, 3, 3]], dtype=float))
    assert w.count == 3
    assert np.allclose(w.bbox_min, [0.0, -2.0, -6.0])
    assert np.allclose(w.bbox_max, [5.0, 4.0, 3.0])


def test_streaming_writer_ignores_empty_chunks(tmp_path):
    p = tmp_path / "e.ply"
    with StreamingPlyWriter(p, with_color=True) as w:
        w.add(np.empty((0, 3)), np.empty((0, 3)))
        w.add(np.array([[1.0, 1.0, 1.0]]), np.array([[9, 9, 9]], dtype=float))
    assert w.count == 1
    r, _ = read_ply_points(p)
    assert len(r) == 1


def test_streaming_writer_no_color_roundtrip(tmp_path):
    p = tmp_path / "nc.ply"
    with StreamingPlyWriter(p, with_color=False) as w:
        w.add(np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
        w.add(np.array([[4.0, 5.0, 6.0]]))
    r, c = read_ply_points(p)
    assert c is None
    assert np.allclose(r, [[0, 0, 0], [1, 2, 3], [4, 5, 6]], atol=1e-5)


def test_streaming_writer_discards_on_exception(tmp_path):
    # A failure mid-fusion must not leave a truncated/bogus PLY behind, and the temp body is cleaned up.
    p = tmp_path / "boom.ply"
    with pytest.raises(RuntimeError, match="boom"), StreamingPlyWriter(p, with_color=True) as w:
        w.add(np.array([[1.0, 1.0, 1.0]]), np.array([[1, 2, 3]], dtype=float))
        raise RuntimeError("boom")
    assert not p.exists()
    assert not (tmp_path / "boom.ply.body.tmp").exists()


def test_iter_ply_chunks_matches_read_with_color(tmp_path):
    # Streaming the cloud in several bounded chunks must reconstruct exactly what read_ply_points
    # returns in one shot — that equivalence is what lets S9/S10 avoid loading the whole cloud.
    rng = np.random.default_rng(0)
    pts = rng.uniform(-10, 10, size=(50, 3))
    cols = rng.integers(0, 256, size=(50, 3)).astype(float)
    p = tmp_path / "chunky.ply"
    write_ply_points(p, pts, cols)

    chunks = list(iter_ply_chunks(p, chunk_points=7))   # 50/7 -> 8 chunks incl. a short final one
    assert len(chunks) == 8
    r = np.vstack([cp for cp, _ in chunks])
    c = np.vstack([cc for _, cc in chunks])
    full_p, full_c = read_ply_points(p)
    assert r.dtype == np.float32 and c.dtype == np.uint8
    assert np.allclose(r, full_p, atol=1e-5)
    assert np.array_equal(c, full_c)


def test_iter_ply_chunks_no_color(tmp_path):
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    p = tmp_path / "nc.ply"
    write_ply_points(p, pts)
    chunks = list(iter_ply_chunks(p, chunk_points=2))
    assert all(c is None for _, c in chunks)
    assert np.allclose(np.vstack([cp for cp, _ in chunks]), pts, atol=1e-5)


def test_iter_ply_chunks_single_chunk_covers_all(tmp_path):
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    cols = np.array([[255, 0, 0], [0, 128, 255]], dtype=float)
    p = tmp_path / "one.ply"
    write_ply_points(p, pts, cols)
    chunks = list(iter_ply_chunks(p, chunk_points=10_000))   # chunk >> n -> exactly one chunk
    assert len(chunks) == 1
    assert len(chunks[0][0]) == 2


def test_float64_ply_roundtrip_preserves_large_crs_coordinates(tmp_path):
    points = np.array([[437021.123456789, 4572795.987654321, 248.123456789]], dtype=np.float64)
    path = tmp_path / "precise.ply"
    write_ply_points(path, points, coordinate_dtype="float64")
    recovered, _ = read_ply_points(path)
    assert recovered.dtype == np.float64
    assert np.allclose(recovered, points, atol=1e-9)
