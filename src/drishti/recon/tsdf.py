"""TSDF fusion via Open3D — real, guarded.

Integrates per-keyframe metric depth + color into a scalable TSDF volume and extracts a dense colored
point cloud. No Open3D installed => loud failure. The fused cloud is the metric dense product S8 meshes.

A single monolithic ``ScalableTSDFVolume`` spanning a large aerial scene at a fine voxel size hashes an
enormous number of surface blocks and can exhaust host RAM (the OS OOM-kills the run). :func:`fuse_tsdf_tiled`
bounds peak memory by partitioning the *ground* into tiles, fusing each tile in its own volume that is
freed before the next, and cropping each tile's points to its core cell so overlapping tiles do not
double-count.

The subtlety, and the fix that actually bounds memory for aerial (nadir) capture: a camera ~100 m AGL sees
a ground footprint far wider than a tile, so simply grouping frames by camera-center XY and integrating
each frame's *full* depth map still paints TSDF blocks across the whole scene — every tile then costs
almost as much as the monolithic volume. So before integrating, each depth map is clipped (via
:func:`_clip_depth_to_box`) to the tile's ground-XY window: a tile's volume only hashes blocks for its own
ground region (+overlap).

Bounding XY is necessary but not sufficient. Monocular depth (Depth Anything V2, made metric by a per-frame
scale+shift fit) is *noisy*, and with a large ``depth_trunc`` the unreliable far field scatters back-projected
points across a thick vertical Z slab — every ground column then hashes many vertical blocks and one tile
alone can exhaust RAM. So the same clip also rejects pixels whose back-projected world-Z falls outside the
scene's true surface envelope (``z_lo..z_hi``), which S7 derives from S2's metric sparse points. That drops
gross far-field flyers — pixels no real surface produced — thinning the slab. It is a quality *gain*, not a
loss: sparse points bracket the true ground and structure, and only depth the geometry never supported is
discarded. Voxel size and the ``depth_trunc`` range are untouched; ``tile_m`` and the Z band together trade
tile count and coverage against peak RAM.
"""
from __future__ import annotations

import gc
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from ..logging import get_logger
from ..runtime.deps import require

log = get_logger("drishti.recon.tsdf")


def _make_volume(o3d, voxel_m: float, sdf_trunc: float):
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_m),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def _integrate(o3d, vol, intr, color_bgr: np.ndarray, depth: np.ndarray,
               extr: np.ndarray, depth_trunc: float) -> None:
    rgb = np.ascontiguousarray(color_bgr[..., ::-1])  # BGR->RGB
    color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32)))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=float(depth_trunc),
        convert_rgb_to_intensity=False,
    )
    vol.integrate(rgbd, intr, extr.astype(np.float64))


def robust_z_band(
    z_values: np.ndarray, margin_m: float, lo_pct: float = 1.0, hi_pct: float = 99.0,
) -> tuple[float, float]:
    """Robust vertical surface envelope ``[z_lo, z_hi]`` from a set of world-Z samples.

    ``z_values`` are the Z coordinates of the metric sparse points (COLMAP-triangulated true surface, in
    the same ground CRS as the poses). Trimming to the ``[lo_pct, hi_pct]`` percentiles discards sparse
    triangulation flyers; ``margin_m`` then pads both ends so legitimate structure (building tops, ground
    dips) and the depth's own noise at the true surface survive, while gross far-field monocular flyers —
    which land far outside this band — are clipped. Empty/all-invalid input returns an open band
    ``(-inf, inf)`` so clipping becomes a no-op (fail-soft: never delete geometry when we cannot bound it).
    """
    z = np.asarray(z_values, dtype=float).ravel()
    z = z[np.isfinite(z)]
    if z.size == 0:
        return (-np.inf, np.inf)
    z_lo = float(np.percentile(z, lo_pct)) - float(margin_m)
    z_hi = float(np.percentile(z, hi_pct)) + float(margin_m)
    return (z_lo, z_hi)


def _clip_depth_to_box(
    depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
    extr: np.ndarray, x_lo: float, x_hi: float, y_lo: float, y_hi: float,
    z_lo: float = -np.inf, z_hi: float = np.inf,
) -> np.ndarray:
    """Return a copy of ``depth`` with every pixel whose back-projected world point falls outside the box
    ``[x_lo, x_hi) x [y_lo, y_hi) x [z_lo, z_hi)`` set to 0 (invalid).

    The XY bounds are what let per-tile memory track tile extent for nadir aerial imagery: each camera sees
    a ground footprint far wider than a tile, so integrating a frame's *full* depth map would paint TSDF
    blocks across the whole scene regardless of which tile owns the camera. Clipping to the tile's ground
    window means a tile's volume only hashes blocks for its own ground region (+overlap).

    The Z bounds bound the *vertical* slab: noisy monocular depth over a large ``depth_trunc`` scatters
    far-field points across many vertical blocks per ground column. Rejecting pixels whose back-projected
    world-Z lands outside the scene's true surface envelope (from S2 sparse points, see
    :func:`robust_z_band`) drops those flyers. Depth range (``depth_trunc``) and voxel size are untouched —
    only pixels no real surface produced, or that another tile owns, are dropped from *this* tile.
    """
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    d = depth.astype(np.float64)
    # camera-frame ray * depth
    xc = (us - cx) / fx * d
    yc = (vs - cy) / fy * d
    pc = np.stack([xc, yc, d], axis=-1)          # (h, w, 3)
    R = extr[:3, :3]
    t = extr[:3, 3]
    # world = R^T (cam - t); for row-vectors that is (pc - t) @ R
    pw = (pc - t) @ R
    inside = (
        (d > 0) & (pw[..., 0] >= x_lo) & (pw[..., 0] < x_hi)
        & (pw[..., 1] >= y_lo) & (pw[..., 1] < y_hi)
        & (pw[..., 2] >= z_lo) & (pw[..., 2] < z_hi)
    )
    out = depth.copy()
    out[~inside] = 0.0
    return out


def depth_world_chunks(depth, extr, fx, fy, cx, cy, depth_trunc, rows_per_chunk=64):
    """Back-project valid metric pixels in bounded strips, keeping world coordinates float64."""
    h, w = depth.shape
    for row in range(0, h, rows_per_chunk):
        d = np.asarray(depth[row:row + rows_per_chunk], dtype=np.float64)
        v, u = np.indices(d.shape)
        v += row
        valid = np.isfinite(d) & (d > 0) & (d < depth_trunc)
        if not valid.any():
            continue
        z = d[valid]
        cam = np.column_stack([(u[valid] - cx) * z / fx, (v[valid] - cy) * z / fy, z])
        yield (cam - extr[:3, 3]) @ extr[:3, :3]


def build_tile_plans(
    frame_specs: Sequence[dict],
    tile_m: float,
    overlap_m: float,
    *,
    surface_points: np.ndarray | None = None,
    z_margin_m: float = 0.0,
    fallback_z_band: tuple[float, float] = (-np.inf, np.inf),
    min_local_surface_points: int = 3,
) -> list[dict]:
    """Plan spatial fusion tiles and their evidence bounds without deleting unsupported geometry.

    A global sparse-point Z envelope is useful as a memory guard, but it is unsafe as the primary
    completeness rule: sparse triangulation can legitimately miss a building or terrain patch.  When
    sparse points are supplied, each tile therefore gets a *local* envelope from points in its expanded
    XY window.  Tiles without enough local sparse support deliberately receive an open Z band so valid
    learned depth and camera support are not turned into a blank block.  The caller records those tiles
    as evidence-limited rather than silently treating them as measured.

    The returned dictionaries are JSON-friendly apart from ``selected`` (a list of integer indices).
    Keeping planning separate from Open3D makes this policy unit-testable and reusable by other dense
    backends.
    """
    if tile_m <= 0.0:
        raise ValueError(f"tile_m must be > 0, got {tile_m}")
    if overlap_m < 0.0:
        raise ValueError(f"overlap_m must be >= 0, got {overlap_m}")
    centers = np.asarray([s["center_xy"] for s in frame_specs], dtype=float)
    if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] != 2:
        raise RuntimeError("TSDF fusion received 0 frames (no valid poses/depth).")

    footprints = [s.get("footprint_xy") for s in frame_specs]
    finite_footprints = [f for f in footprints if f is not None]
    mins = np.min([f[:2] for f in finite_footprints], axis=0) if finite_footprints else centers.min(axis=0)
    maxs = np.max([f[2:] for f in finite_footprints], axis=0) if finite_footprints else centers.max(axis=0)
    nx = max(1, int(np.ceil((maxs[0] - mins[0]) / tile_m)))
    ny = max(1, int(np.ceil((maxs[1] - mins[1]) / tile_m)))
    sparse = None
    if surface_points is not None:
        candidate = np.asarray(surface_points, dtype=float)
        if candidate.ndim == 2 and candidate.shape[1] >= 3:
            finite = np.isfinite(candidate[:, :3]).all(axis=1)
            sparse = candidate[finite, :3]

    plans: list[dict] = []
    for iy in range(ny):
        for ix in range(nx):
            x0 = float(mins[0] + ix * tile_m)
            y0 = float(mins[1] + iy * tile_m)
            bounded = bool(finite_footprints)
            core = {
                "x_lo": x0 if bounded or ix > 0 else -np.inf,
                "x_hi": x0 + tile_m if bounded or ix < nx - 1 else np.inf,
                "y_lo": y0 if bounded or iy > 0 else -np.inf,
                "y_hi": y0 + tile_m if bounded or iy < ny - 1 else np.inf,
            }
            window = {
                "x_lo": x0 - overlap_m if bounded or ix > 0 else -np.inf,
                "x_hi": x0 + tile_m + overlap_m if bounded or ix < nx - 1 else np.inf,
                "y_lo": y0 - overlap_m if bounded or iy > 0 else -np.inf,
                "y_hi": y0 + tile_m + overlap_m if bounded or iy < ny - 1 else np.inf,
            }
            selected = np.where(
                (centers[:, 0] >= x0 - overlap_m)
                & (centers[:, 0] < x0 + tile_m + overlap_m)
                & (centers[:, 1] >= y0 - overlap_m)
                & (centers[:, 1] < y0 + tile_m + overlap_m)
            )[0].astype(int).tolist()
            if bounded:
                selected = [i for i, f in enumerate(footprints) if f is not None
                            and f[0] < window["x_hi"] and f[2] >= window["x_lo"]
                            and f[1] < window["y_hi"] and f[3] >= window["y_lo"]]

            z_band = fallback_z_band
            z_source = "global_fallback"
            local_count = 0
            if sparse is not None:
                local = sparse[
                    (sparse[:, 0] >= window["x_lo"])
                    & (sparse[:, 0] < window["x_hi"])
                    & (sparse[:, 1] >= window["y_lo"])
                    & (sparse[:, 1] < window["y_hi"])
                ]
                local_count = int(len(local))
                if local_count >= min_local_surface_points:
                    z_band = robust_z_band(local[:, 2], margin_m=z_margin_m)
                    z_source = "local_sparse"
                else:
                    # This is intentional: missing sparse support is not proof that the tile is empty.
                    z_band = (-np.inf, np.inf)
                    z_source = "open_no_local_sparse"

            plans.append({
                "ix": ix, "iy": iy, "core": core, "window": window,
                "selected": selected, "z_lo": float(z_band[0]), "z_hi": float(z_band[1]),
                "z_source": z_source, "local_sparse_points": local_count,
                "camera_support_method": "metric_depth_footprint" if bounded else "legacy_center_proxy",
            })
    return plans


def fuse_tsdf(
    frames: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    fx: float, fy: float, cx: float, cy: float, width: int, height: int,
    voxel_m: float, sdf_trunc: float, depth_trunc: float,
):
    """Fuse (color_bgr, depth_m, extrinsic_world2cam) frames into a dense point cloud.

    Single-volume path. Returns (points Nx3 float64, colors Nx3 float64 in [0,1], n_frames_used).
    For large aerial scenes prefer :func:`fuse_tsdf_tiled`, which bounds peak RAM.
    """
    o3d = require("open3d", purpose="TSDF fusion")
    intr = o3d.camera.PinholeCameraIntrinsic(int(width), int(height), fx, fy, cx, cy)
    vol = _make_volume(o3d, voxel_m, sdf_trunc)
    used = 0
    for color_bgr, depth, extr in frames:
        if depth is None or color_bgr is None:
            continue
        _integrate(o3d, vol, intr, color_bgr, depth, extr, depth_trunc)
        used += 1
    if used == 0:
        raise RuntimeError("TSDF fusion integrated 0 frames (no valid depth).")
    pcd = vol.extract_point_cloud()
    return np.asarray(pcd.points), np.asarray(pcd.colors), used


def fuse_tsdf_tiled(
    frame_specs: Sequence[dict],
    load: Callable[[dict], tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    fx: float, fy: float, cx: float, cy: float, width: int, height: int,
    voxel_m: float, sdf_trunc: float, depth_trunc: float,
    tile_m: float, overlap_m: float,
    z_lo: float = -np.inf, z_hi: float = np.inf,
    sink: Callable[[np.ndarray, np.ndarray], None] | None = None,
    surface_points: np.ndarray | None = None,
    z_margin_m: float = 0.0,
    min_local_surface_points: int = 3,
    tile_observer: Callable[[dict], None] | None = None,
    max_vertical_slab_m: float = 10.0,
):
    """Memory-bounded TSDF fusion by spatial tiling of the ground footprint.

    ``frame_specs`` is a re-iterable sequence of descriptors, each carrying ``center_xy`` (the camera
    position in the ground CRS, an (x, y) pair) plus whatever ``load`` needs to lazily produce that
    frame's ``(color_bgr, depth_m, extrinsic_world2cam)`` — or ``None`` to skip it. The ground is split
    into ``tile_m``-sized core cells; each cell is fused from the frames whose camera center falls within
    ``overlap_m`` of it, and — critically for wide-footprint nadir imagery — each frame's depth map is
    clipped to the cell's ground window (core + overlap) in XY *and* to the scene's true surface envelope
    ``[z_lo, z_hi]`` in Z before integration, so the tile's volume only hashes blocks for its own ground
    region and only where a real surface exists (see :func:`_clip_depth_to_box`). When ``surface_points``
    are supplied, each tile uses a local sparse-point envelope; tiles without enough local support keep
    an open Z band so valid learned depth cannot be silently turned into a blank block. The extracted points are
    then cropped to the core cell so neighbouring tiles do not double-count. Boundary cells extend to
    +/-inf in XY so nothing is dropped at the scene edge; ``z_lo``/``z_hi`` default to an open band
    (no vertical clipping) when the caller cannot supply a surface envelope.

    Memory: each tile's volume is freed before the next, so peak RAM tracks a single tile — *except* that
    accumulating every tile's extracted points to concatenate at the end costs RAM proportional to the
    whole cloud (100M+ points on a large aerial scene => OOM). Pass ``sink`` to avoid that: it is called
    ``sink(points, colors)`` once per non-empty tile with that tile's core-cropped points, which are then
    freed instead of retained (e.g. a :class:`~drishti.io.pointcloud.StreamingPlyWriter` that streams them
    to disk). With a sink the returned point/color arrays are empty (the sink owns the data); with
    ``sink=None`` the points are accumulated and returned in full (convenient for small clouds and tests).

    Returns (points Nx3 float64, colors Nx3 float64 in [0,1], n_frames_used) — ``n_frames_used`` is the
    count of distinct frames integrated into at least one tile. ``points``/``colors`` are empty when a
    ``sink`` consumed the tiles.
    """
    if any(spec.get("footprint_xy") is not None for spec in frame_specs):
        from .adaptive import fuse_metric_tiles

        # Retry only a completed, empty adaptive result. Streaming returns empty arrays even on
        # success, so count actual sink writes; progress events can repeat cumulative tile counts.
        adaptive_emitted = 0

        def observe_adaptive(event: dict) -> None:
            if tile_observer is not None:
                tile_observer({**event, "fusion_backend": "adaptive_metric"})

        def emit_adaptive(points: np.ndarray, colors: np.ndarray) -> None:
            nonlocal adaptive_emitted
            if sink is not None:
                sink(points, colors)
                adaptive_emitted += len(points)

        adaptive_result = fuse_metric_tiles(
            frame_specs, load, fx, fy, cx, cy, width, height, voxel_m=voxel_m,
            sdf_trunc=sdf_trunc, depth_trunc=depth_trunc, tile_m=tile_m, overlap_m=overlap_m,
            surface_points=surface_points, z_margin_m=z_margin_m,
            min_local_surface_points=min_local_surface_points,
            sink=emit_adaptive if sink is not None else None, tile_observer=observe_adaptive,
            max_vertical_slab_m=max_vertical_slab_m,
        )
        if adaptive_emitted or len(adaptive_result[0]):
            return adaptive_result
        log.warning(
            "adaptive metric TSDF completed with no emitted points; "
            "retrying legacy camera-center tiling"
        )

        def observe_legacy(event: dict) -> None:
            if tile_observer is not None:
                tile_observer({
                    **event,
                    "fusion_backend": "legacy_center",
                    "fallback_reason": "adaptive_empty_cloud",
                })
        # Do not let the retry re-enter footprint-based planning.  The legacy path must use camera
        # centers for both its scene extent and frame membership; otherwise this fallback would only
        # bypass adaptive Z slabs while retaining the same unproven footprint contract.
        legacy_frame_specs = [
            {key: value for key, value in spec.items() if key != "footprint_xy"}
            for spec in frame_specs
        ]
    else:
        observe_legacy = tile_observer
        legacy_frame_specs = frame_specs
    o3d = require("open3d", purpose="TSDF fusion")
    intr = o3d.camera.PinholeCameraIntrinsic(int(width), int(height), fx, fy, cx, cy)

    plans = build_tile_plans(
        legacy_frame_specs, tile_m, overlap_m, surface_points=surface_points,
        z_margin_m=z_margin_m, fallback_z_band=(z_lo, z_hi),
        min_local_surface_points=min_local_surface_points,
    )
    centers = np.asarray([s["center_xy"] for s in legacy_frame_specs], dtype=float)
    mins = centers.min(axis=0)
    maxs = centers.max(axis=0)
    nx = max(p["ix"] for p in plans) + 1
    ny = max(p["iy"] for p in plans) + 1
    z_desc = "adaptive-local" if surface_points is not None else (
        "open" if not (np.isfinite(z_lo) and np.isfinite(z_hi))
        else f"{z_lo:.1f}..{z_hi:.1f}m ({z_hi - z_lo:.1f}m)"
    )
    log.info(
        "TSDF tiled fusion: %d frames over %.0f x %.0f m -> %d x %d = %d tiles "
        "(tile_m=%.0f, overlap_m=%.0f, voxel_m=%.3f, depth_trunc=%.0f, z_band=%s)",
        len(centers), maxs[0] - mins[0], maxs[1] - mins[1], nx, ny, nx * ny,
        tile_m, overlap_m, voxel_m, depth_trunc, z_desc,
    )

    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    used_idx: set[int] = set()

    for plan in plans:
            ix, iy = plan["ix"], plan["iy"]
            cx_lo, cx_hi = plan["core"]["x_lo"], plan["core"]["x_hi"]
            cy_lo, cy_hi = plan["core"]["y_lo"], plan["core"]["y_hi"]
            ix_lo, ix_hi = plan["window"]["x_lo"], plan["window"]["x_hi"]
            iy_lo, iy_hi = plan["window"]["y_lo"], plan["window"]["y_hi"]
            sel = np.asarray(plan["selected"], dtype=int)
            if len(sel) == 0:
                if observe_legacy is not None:
                    observe_legacy({
                        "tile_id": f"{ix}:{iy}", "ix": ix, "iy": iy,
                        "candidate_frames": 0, "frames_integrated": 0,
                        "points_extracted": 0, "points_emitted": 0,
                        "geometry_class": "unknown", "status": "unobservable",
                        "reason": "no_camera_support", "confidence": 0.0,
                        "z_band": [plan["z_lo"], plan["z_hi"]],
                        "z_source": plan["z_source"],
                        "local_sparse_points": plan["local_sparse_points"],
                    })
                continue

            # Logged before integration so an OOM inside a tile is attributable to a specific tile
            # (the post-extract summary below never prints if the volume exhausts RAM first).
            log.info("  tile [%d,%d] integrating %d candidate frames (z=%s)...", ix, iy,
                     len(sel), plan["z_source"])
            vol = _make_volume(o3d, voxel_m, sdf_trunc)
            n_here = 0
            for i in sel:
                loaded = load(frame_specs[int(i)])
                if loaded is None:
                    continue
                color_bgr, depth, extr = loaded
                if depth is None or color_bgr is None:
                    continue
                depth_clipped = _clip_depth_to_box(
                    depth, fx, fy, cx, cy, extr, ix_lo, ix_hi, iy_lo, iy_hi,
                    plan["z_lo"], plan["z_hi"],
                )
                if not np.any(depth_clipped > 0):
                    continue  # this frame sees no ground inside the tile window
                _integrate(o3d, vol, intr, color_bgr, depth_clipped, extr, depth_trunc)
                used_idx.add(int(i))
                n_here += 1
            if n_here == 0:
                del vol
                gc.collect()
                if observe_legacy is not None:
                    observe_legacy({
                        "tile_id": f"{ix}:{iy}", "ix": ix, "iy": iy,
                        "candidate_frames": len(sel), "frames_integrated": 0,
                        "points_extracted": 0, "points_emitted": 0,
                        "geometry_class": "unknown", "status": "unobservable",
                        "reason": "no_valid_depth_support", "confidence": 0.0,
                        "z_band": [plan["z_lo"], plan["z_hi"]],
                        "z_source": plan["z_source"],
                        "local_sparse_points": plan["local_sparse_points"],
                    })
                continue

            pcd = vol.extract_point_cloud()
            pts = np.asarray(pcd.points)
            cols = np.asarray(pcd.colors)
            del vol, pcd
            gc.collect()
            if len(pts) == 0:
                if observe_legacy is not None:
                    observe_legacy({
                        "tile_id": f"{ix}:{iy}", "ix": ix, "iy": iy,
                        "candidate_frames": len(sel), "frames_integrated": n_here,
                        "points_extracted": 0, "points_emitted": 0,
                        "geometry_class": "unknown", "status": "unobservable",
                        "reason": "fusion_empty",
                        "confidence": round(n_here / max(1, len(sel)), 4),
                        "z_band": [plan["z_lo"], plan["z_hi"]],
                        "z_source": plan["z_source"],
                        "local_sparse_points": plan["local_sparse_points"],
                    })
                continue

            # crop to the core cell so overlapping tiles don't emit duplicate points
            keep = (
                (pts[:, 0] >= cx_lo) & (pts[:, 0] < cx_hi)
                & (pts[:, 1] >= cy_lo) & (pts[:, 1] < cy_hi)
            )
            n_keep = int(keep.sum())
            log.info("  tile [%d,%d] fused %d frames -> %d pts (%d after core-crop)",
                     ix, iy, n_here, len(pts), n_keep)
            if keep.any():
                if sink is not None:
                    # Stream this tile's points to the sink and free them: peak RAM tracks one tile,
                    # not the whole (100M+ point) cloud. The sink owns the data from here.
                    sink(pts[keep], cols[keep])
                else:
                    all_pts.append(pts[keep])
                    all_cols.append(cols[keep])
            if observe_legacy is not None:
                observe_legacy({
                    "tile_id": f"{ix}:{iy}", "ix": ix, "iy": iy,
                    "candidate_frames": len(sel), "frames_integrated": n_here,
                    "points_extracted": int(len(pts)), "points_emitted": int(n_keep),
                    "geometry_class": "reconstructed" if n_keep else "unknown",
                    "status": "reconstructed" if n_keep else "unobservable",
                    "reason": None if n_keep else "no_surface_in_core",
                    "confidence": round(n_here / max(1, len(sel)), 4) if n_keep else 0.0,
                    "z_band": [plan["z_lo"], plan["z_hi"]],
                    "z_source": plan["z_source"],
                    "local_sparse_points": plan["local_sparse_points"],
                })
            del pts, cols
            gc.collect()

    if not used_idx:
        raise RuntimeError("TSDF fusion integrated 0 frames (no valid depth).")
    if not all_pts:
        return (np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float), len(used_idx))
    return np.concatenate(all_pts, axis=0), np.concatenate(all_cols, axis=0), len(used_idx)
