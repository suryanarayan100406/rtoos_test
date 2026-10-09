"""Tiled TSDF fusion: partition math, core-cell cropping, and dedup.

Open3D is an optional native dep and is usually absent in CI, so we inject a minimal fake ``open3d``
module. The fake volume simply remembers every point handed to it (one point per integrated frame, at
that frame's camera XY) and returns them from ``extract_point_cloud`` — enough to exercise the tiling,
overlap selection, and boundary-inclusive cropping that :func:`fuse_tsdf_tiled` owns.
"""
import sys
import types

import numpy as np
import pytest


class _FakeImage:
    def __init__(self, arr):
        self.arr = arr


class _FakeRGBD:
    def __init__(self, color, depth):
        self.color, self.depth = color, depth


class _FakePCD:
    def __init__(self, pts):
        self.points = pts
        self.colors = np.tile(np.array([0.5, 0.5, 0.5]), (len(pts), 1))


class _FakeVolume:
    """Records the world XY of each integrated camera; emits one point per camera at that XY."""

    def __init__(self, **kwargs):
        self._cams = []

    def integrate(self, rgbd, intr, extr):
        # extr is world->cam (R | -R c). Recover camera center c = -R^T t.
        R = np.asarray(extr)[:3, :3]
        t = np.asarray(extr)[:3, 3]
        c = -R.T @ t
        self._cams.append(c)

    def extract_point_cloud(self):
        return _FakePCD(np.array(self._cams, dtype=float) if self._cams else np.empty((0, 3)))


def _install_fake_open3d(monkeypatch):
    o3d = types.ModuleType("open3d")
    o3d.camera = types.SimpleNamespace(
        PinholeCameraIntrinsic=lambda *a, **k: object()
    )
    o3d.geometry = types.SimpleNamespace(
        Image=_FakeImage,
        RGBDImage=types.SimpleNamespace(
            create_from_color_and_depth=lambda color, depth, **k: _FakeRGBD(color, depth)
        ),
    )
    o3d.pipelines = types.SimpleNamespace(
        integration=types.SimpleNamespace(
            ScalableTSDFVolume=_FakeVolume,
            TSDFVolumeColorType=types.SimpleNamespace(RGB8=0),
        )
    )
    monkeypatch.setitem(sys.modules, "open3d", o3d)


def _spec(x, y):
    R = np.eye(3)
    c = np.array([x, y, 10.0])
    extr = np.eye(4)
    extr[:3, :3] = R
    extr[:3, 3] = -R @ c
    return {"center_xy": (x, y), "extr": extr, "id": (x, y)}


def _load(spec):
    color = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float32)
    return color, depth, spec["extr"]


def test_tiled_fusion_covers_all_and_no_duplicates(monkeypatch):
    from drishti.recon.tsdf import fuse_tsdf_tiled

    _install_fake_open3d(monkeypatch)
    # 5x5 grid of cameras 20 m apart spanning 0..80 m; 30 m tiles => several core cells.
    specs = [_spec(20.0 * i, 20.0 * j) for i in range(5) for j in range(5)]

    pts, cols, used = fuse_tsdf_tiled(
        specs, _load, fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4,
        voxel_m=0.05, sdf_trunc=0.2, depth_trunc=150.0,
        tile_m=30.0, overlap_m=8.0,
    )

    assert used == 25
    # Each camera-center point lands in exactly one core cell => exactly 25 points, no duplicates.
    assert len(pts) == 25
    assert len(cols) == 25
    xy = {(round(p[0], 3), round(p[1], 3)) for p in pts}
    assert xy == {(20.0 * i, 20.0 * j) for i in range(5) for j in range(5)}


def test_tiled_keeps_far_field_at_scene_edge(monkeypatch):
    from drishti.recon.tsdf import fuse_tsdf_tiled

    _install_fake_open3d(monkeypatch)
    # Two clusters far apart on X: the boundary cells extend to +/-inf so nothing is dropped.
    specs = [_spec(0.0, 0.0), _spec(5.0, 0.0), _spec(500.0, 0.0)]
    pts, _, used = fuse_tsdf_tiled(
        specs, _load, fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4,
        voxel_m=0.05, sdf_trunc=0.2, depth_trunc=150.0,
        tile_m=30.0, overlap_m=8.0,
    )
    assert used == 3
    xs = sorted(round(p[0], 1) for p in pts)
    assert xs == [0.0, 5.0, 500.0]


def test_tiled_raises_on_no_frames(monkeypatch):
    from drishti.recon.tsdf import fuse_tsdf_tiled

    _install_fake_open3d(monkeypatch)
    with pytest.raises(RuntimeError, match="0 frames"):
        fuse_tsdf_tiled(
            [], _load, fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4,
            voxel_m=0.05, sdf_trunc=0.2, depth_trunc=150.0, tile_m=30.0, overlap_m=8.0,
        )


def test_tiled_skips_frames_load_returns_none(monkeypatch):
    from drishti.recon.tsdf import fuse_tsdf_tiled

    _install_fake_open3d(monkeypatch)
    specs = [_spec(0.0, 0.0), _spec(10.0, 0.0), _spec(20.0, 0.0)]

    def _load_some_none(spec):
        if spec["id"] == (10.0, 0.0):
            return None
        return _load(spec)

    pts, _, used = fuse_tsdf_tiled(
        specs, _load_some_none, fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4,
        voxel_m=0.05, sdf_trunc=0.2, depth_trunc=150.0, tile_m=30.0, overlap_m=8.0,
    )
    assert used == 2
    assert len(pts) == 2


def _nadir_extr(cx_world, cy_world, height):
    """world->cam for a nadir camera at (cx_world, cy_world, height) looking straight down (-Z world).

    Camera axes: +Xc = +X world, +Yc = -Y world, +Zc = -Z world (so a positive camera-frame depth maps
    to a point *below* the camera). extr maps world->cam: p_cam = R (p_world - c).
    """
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0]])
    c = np.array([cx_world, cy_world, height])
    extr = np.eye(4)
    extr[:3, :3] = R
    extr[:3, 3] = -R @ c
    return extr


def test_clip_depth_to_box_keeps_only_pixels_inside_window():
    from drishti.recon.tsdf import _clip_depth_to_box

    # 5x5 nadir view, camera 100 m above world origin, 1 m/px ground sampling at this depth.
    fx = fy = 100.0
    cxp = cyp = 2.0  # principal point at pixel (2,2)
    h = w = 5
    depth = np.full((h, w), 100.0, dtype=np.float32)  # flat ground 100 m below
    extr = _nadir_extr(0.0, 0.0, 100.0)

    # Ground XY under each pixel: X = (u-2)/100 * 100 = u-2  in [-2..2]; Y = -(v-2) in [-2..2].
    # Keep the window [0,3) x [0,3): X in {0,1,2} -> u in {2,3,4}; Y in {0,1,2} -> v in {0,1,2}.
    out = _clip_depth_to_box(depth, fx, fy, cxp, cyp, extr,
                            x_lo=0.0, x_hi=3.0, y_lo=0.0, y_hi=3.0)
    kept = out > 0
    expected = np.zeros((h, w), dtype=bool)
    for v in range(h):
        for u in range(w):
            X = (u - cxp)
            Y = -(v - cyp)
            expected[v, u] = (0.0 <= X < 3.0) and (0.0 <= Y < 3.0)
    assert np.array_equal(kept, expected)
    # kept pixels keep their original depth value; others are zeroed
    assert np.all(out[kept] == 100.0)
    assert np.all(out[~kept] == 0.0)


def test_clip_depth_to_box_ignores_invalid_depth():
    from drishti.recon.tsdf import _clip_depth_to_box

    fx = fy = 100.0
    depth = np.zeros((5, 5), dtype=np.float32)  # all invalid
    extr = _nadir_extr(0.0, 0.0, 100.0)
    out = _clip_depth_to_box(depth, fx, fy, 2.0, 2.0, extr,
                            x_lo=-np.inf, x_hi=np.inf, y_lo=-np.inf, y_hi=np.inf)
    assert np.all(out == 0.0)  # zero-depth pixels never become valid, even in an infinite window


def test_clip_depth_to_box_infinite_window_is_noop():
    from drishti.recon.tsdf import _clip_depth_to_box

    fx = fy = 100.0
    depth = np.full((5, 5), 100.0, dtype=np.float32)
    extr = _nadir_extr(50.0, -30.0, 100.0)
    out = _clip_depth_to_box(depth, fx, fy, 2.0, 2.0, extr,
                            x_lo=-np.inf, x_hi=np.inf, y_lo=-np.inf, y_hi=np.inf)
    assert np.array_equal(out, depth)  # boundary tiles (+/-inf on all sides) drop nothing


def test_clip_depth_to_box_z_band_keeps_only_pixels_in_band():
    from drishti.recon.tsdf import _clip_depth_to_box

    # Nadir camera 100 m up: a pixel with depth d back-projects to world Z = 100 - d. Make row v have
    # depth 100 - v so world Z == v (rows 0..4 -> Z 0..4). A Z band [1.5, 3.5) must keep exactly rows 2,3.
    fx = fy = 100.0
    h = w = 5
    depth = np.empty((h, w), dtype=np.float32)
    for v in range(h):
        depth[v, :] = 100.0 - v
    extr = _nadir_extr(0.0, 0.0, 100.0)
    out = _clip_depth_to_box(depth, fx, fy, 2.0, 2.0, extr,
                             x_lo=-np.inf, x_hi=np.inf, y_lo=-np.inf, y_hi=np.inf,
                             z_lo=1.5, z_hi=3.5)
    kept_rows = {v for v in range(h) if np.any(out[v] > 0)}
    assert kept_rows == {2, 3}
    # within a kept row every pixel survives (flat depth across the row, XY unbounded)
    assert np.all(out[2] == depth[2]) and np.all(out[3] == depth[3])
    assert np.all(out[0] == 0.0) and np.all(out[1] == 0.0) and np.all(out[4] == 0.0)


def test_clip_depth_to_box_z_band_drops_far_field_flyer():
    from drishti.recon.tsdf import _clip_depth_to_box

    # One noisy far pixel (depth 145 -> world Z = -45) among ground pixels (depth 100 -> Z 0); a band
    # around ground [-5, 5) drops the flyer and keeps the ground, the whole point of the vertical clip.
    fx = fy = 100.0
    depth = np.full((5, 5), 100.0, dtype=np.float32)
    depth[0, 0] = 145.0
    extr = _nadir_extr(0.0, 0.0, 100.0)
    out = _clip_depth_to_box(depth, fx, fy, 2.0, 2.0, extr,
                             x_lo=-np.inf, x_hi=np.inf, y_lo=-np.inf, y_hi=np.inf,
                             z_lo=-5.0, z_hi=5.0)
    assert out[0, 0] == 0.0                 # far flyer removed
    assert np.count_nonzero(out) == 24      # the other 24 ground pixels kept


def test_robust_z_band_trims_flyers_and_pads_margin():
    from drishti.recon.tsdf import robust_z_band

    # 200 samples on a flat surface at Z=0 plus one triangulation flyer at 500 m: p1/p99 exclude the
    # flyer, so the band is just [0-margin, 0+margin] and the flyer is far outside it.
    z = np.concatenate([np.zeros(200), np.array([500.0])])
    z_lo, z_hi = robust_z_band(z, margin_m=2.0)
    assert z_lo == pytest.approx(-2.0, abs=0.5)
    assert z_hi == pytest.approx(2.0, abs=0.5)
    assert z_hi < 500.0

    # Real relief must survive: a 0..20 m spread pads out beyond both ends by the margin.
    z2 = np.linspace(0.0, 20.0, 100)
    lo2, hi2 = robust_z_band(z2, margin_m=3.0)
    assert lo2 < 0.0 and hi2 > 20.0


def test_robust_z_band_empty_or_invalid_is_open():
    from drishti.recon.tsdf import robust_z_band

    assert robust_z_band(np.empty((0,)), margin_m=5.0) == (-np.inf, np.inf)
    assert robust_z_band(np.array([np.nan, np.inf, -np.inf]), margin_m=5.0) == (-np.inf, np.inf)


def test_tiled_fusion_threads_z_band_through_clip(monkeypatch):
    """A tight Z band that excludes the (fake) integrated points drops everything; an open band keeps
    them — proving fuse_tsdf_tiled forwards z_lo/z_hi into the per-frame clip."""
    from drishti.recon.tsdf import fuse_tsdf_tiled

    _install_fake_open3d(monkeypatch)

    # Nadir cameras at height 10; _load returns flat depth 1.0 => world Z = 10 - 1 = 9 for every pixel.
    def _nadir_spec(x, y):
        extr = _nadir_extr(x, y, 10.0)
        return {"center_xy": (x, y), "extr": extr, "id": (x, y)}

    def _nadir_load(spec):
        return np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32), spec["extr"]

    specs = [_nadir_spec(10.0 * i, 0.0) for i in range(3)]

    # Band around Z=9 keeps all frames.
    _, _, used_open = fuse_tsdf_tiled(
        specs, _nadir_load, fx=100.0, fy=100.0, cx=2.0, cy=2.0, width=4, height=4,
        voxel_m=0.05, sdf_trunc=0.2, depth_trunc=150.0, tile_m=30.0, overlap_m=8.0,
        z_lo=5.0, z_hi=15.0,
    )
    assert used_open == 3
    # Band far from Z=9 clips every pixel of every frame => 0 frames integrated.
    with pytest.raises(RuntimeError, match="0 frames"):
        fuse_tsdf_tiled(
            specs, _nadir_load, fx=100.0, fy=100.0, cx=2.0, cy=2.0, width=4, height=4,
            voxel_m=0.05, sdf_trunc=0.2, depth_trunc=150.0, tile_m=30.0, overlap_m=8.0,
            z_lo=100.0, z_hi=200.0,
        )


def test_metric_footprint_dispatches_to_adaptive_fusion(monkeypatch):
    from drishti.recon.tsdf import fuse_tsdf_tiled

    _install_fake_open3d(monkeypatch)

    class _DepthSurfaceVolume(_FakeVolume):
        def integrate(self, rgbd, intr, extr):
            R = np.asarray(extr)[:3, :3]
            t = np.asarray(extr)[:3, 3]
            c = -R.T @ t
            self._cams.append(c + np.array([0.0, 0.0, -1.0]))

    sys.modules["open3d"].pipelines.integration.ScalableTSDFVolume = _DepthSurfaceVolume

    def _nadir_spec(x, y):
        extr = _nadir_extr(x, y, 10.0)
        return {"center_xy": (x, y), "extr": extr, "id": (x, y),
                "footprint_xy": [-2.0, -2.0, 2.0, 2.0]}

    def _nadir_load(spec):
        return np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32), spec["extr"]

    events = []
    points, colors, used = fuse_tsdf_tiled(
        [_nadir_spec(0.0, 0.0)], _nadir_load, fx=100.0, fy=100.0, cx=2.0, cy=2.0,
        width=4, height=4, voxel_m=0.05, sdf_trunc=0.2, depth_trunc=150.0,
        tile_m=10.0, overlap_m=1.0, tile_observer=events.append,
    )
    assert used == 1
    assert len(points) > 0
    assert len(colors) == len(points)
    assert events and events[0]["status"] == "reconstructed"


def test_tiled_fusion_sink_streams_tiles_and_returns_empty(monkeypatch):
    """With a sink, fuse_tsdf_tiled must hand every kept point to the sink (freeing per tile) and return
    empty arrays — the RAM-bounded S7 path — while covering exactly the same points as the accumulating
    path. This is what keeps peak RAM at one tile instead of the whole (100M+ point) cloud."""
    from drishti.recon.tsdf import fuse_tsdf_tiled

    _install_fake_open3d(monkeypatch)
    specs = [_spec(20.0 * i, 20.0 * j) for i in range(5) for j in range(5)]
    kw = {"fx": 1.0, "fy": 1.0, "cx": 2.0, "cy": 2.0, "width": 4, "height": 4,
          "voxel_m": 0.05, "sdf_trunc": 0.2, "depth_trunc": 150.0, "tile_m": 30.0, "overlap_m": 8.0}

    # Reference: accumulate-and-return (sink=None).
    ref_pts, ref_cols, ref_used = fuse_tsdf_tiled(specs, _load, **kw)

    # Streamed: collect via a sink.
    chunks_pts: list[np.ndarray] = []
    chunks_cols: list[np.ndarray] = []

    def _sink(pts, cols):
        chunks_pts.append(np.array(pts))   # copy: the function frees its buffers after the call
        chunks_cols.append(np.array(cols))

    out_pts, out_cols, used = fuse_tsdf_tiled(specs, _load, sink=_sink, **kw)

    assert used == ref_used == 25
    assert len(out_pts) == 0 and len(out_cols) == 0     # sink owns the data; nothing accumulated
    streamed = np.concatenate(chunks_pts, axis=0)
    assert len(streamed) == len(ref_pts) == 25
    # Same set of points regardless of tile emission order.
    ref_xy = {(round(p[0], 3), round(p[1], 3)) for p in ref_pts}
    got_xy = {(round(p[0], 3), round(p[1], 3)) for p in streamed}
    assert got_xy == ref_xy
    assert sum(len(c) for c in chunks_cols) == 25


def test_tile_plan_opens_z_band_when_sparse_support_is_missing():
    """Sparse coverage gaps must not be converted into a silent blank tile."""
    from drishti.recon.tsdf import build_tile_plans

    specs = [_spec(0.0, 0.0), _spec(60.0, 0.0)]
    plans = build_tile_plans(
        specs, tile_m=30.0, overlap_m=0.0,
        surface_points=np.array([[60.0, 0.0, 100.0]]),
        z_margin_m=1.0, fallback_z_band=(90.0, 110.0), min_local_surface_points=2,
    )
    first = next(p for p in plans if p["ix"] == 0)
    assert first["z_source"] == "open_no_local_sparse"
    assert first["z_lo"] == -np.inf and first["z_hi"] == np.inf


def test_valid_depth_with_sparse_gap_is_reconstructed_not_blank(monkeypatch):
    """Regression: valid learned depth + camera support survives a sparse-point coverage gap."""
    from drishti.recon.tsdf import fuse_tsdf_tiled

    _install_fake_open3d(monkeypatch)
    specs = [_spec(0.0, 0.0), _spec(60.0, 0.0)]
    events = []
    # The sparse point is only in the second tile. The first tile has valid depth at Z=9, but a global
    # fallback band [100, 101] that would have deleted it in the old implementation.
    _, _, used = fuse_tsdf_tiled(
        specs, _load, fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4,
        voxel_m=0.05, sdf_trunc=0.2, depth_trunc=150.0, tile_m=30.0, overlap_m=0.0,
        z_lo=100.0, z_hi=101.0, surface_points=np.array([[60.0, 0.0, 100.0]]),
        min_local_surface_points=2, tile_observer=events.append,
    )
    assert used >= 1
    first = next(event for event in events if event["ix"] == 0 and event["iy"] == 0)
    assert first["status"] == "reconstructed"
    assert first["geometry_class"] == "reconstructed"


def test_tile_observer_explains_unobservable_gap(monkeypatch):
    """A true camera-support gap is explicit and machine-readable, never an unexplained empty block."""
    from drishti.recon.tsdf import fuse_tsdf_tiled

    _install_fake_open3d(monkeypatch)
    specs = [_spec(0.0, 0.0), _spec(90.0, 0.0)]
    events = []
    fuse_tsdf_tiled(
        specs, _load, fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4,
        voxel_m=0.05, sdf_trunc=0.2, depth_trunc=150.0, tile_m=30.0, overlap_m=0.0,
        tile_observer=events.append,
    )
    gap = next(event for event in events if event["ix"] == 1 and event["iy"] == 0)
    assert gap["status"] == "unobservable"
    assert gap["reason"] == "no_camera_support"
    assert gap["geometry_class"] == "unknown"
