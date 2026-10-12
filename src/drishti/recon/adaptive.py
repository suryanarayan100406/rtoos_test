"""Depth-frustum-driven spatial fusion with bounded vertical slabs, preserving learned coverage."""
from __future__ import annotations

import gc

import numpy as np

from ..logging import get_logger
from ..runtime.deps import require
from .camera import recenter_extrinsic
from .tsdf import (
    _clip_depth_to_box,
    _integrate,
    _make_volume,
    build_tile_plans,
    depth_world_chunks,
    tile_origin_from_specs,
)


def fuse_metric_tiles(frame_specs, load, fx, fy, cx, cy, width, height, *, voxel_m,
                      sdf_trunc, depth_trunc, tile_m, overlap_m, surface_points,
                      z_margin_m, min_local_surface_points, sink, tile_observer,
                      max_vertical_slab_m):
    """Allocate one finite XY/Z working volume at a time; sparse geometry is evidence, not a veto.

    Surface extent and frame membership come from metric depth footprints. A streamed prepass within
    each tile finds the complete valid depth envelope, including surfaces missed by sparse geometry.
    A wide vertical envelope is split into overlapping slabs rather than clipped away. Slab outputs
    are cropped to disjoint XYZ cores. Evidence counts describe support, never an accuracy probability.
    """
    if max_vertical_slab_m <= 0:
        raise ValueError("max_vertical_slab_m must be positive")
    o3d = require("open3d", purpose="adaptive metric TSDF fusion")
    intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    plans = build_tile_plans(frame_specs, tile_m, overlap_m, surface_points=surface_points,
                             z_margin_m=z_margin_m, min_local_surface_points=min_local_surface_points)
    all_points, all_colors, used = [], [], set()
    log = get_logger("drishti.recon.adaptive")
    for plan in plans:
        window, core, sel = plan["window"], plan["core"], plan["selected"]
        event = {
            "tile_id": f"{plan['ix']}:{plan['iy']}", "ix": plan["ix"], "iy": plan["iy"],
            "bounds_xy": [core["x_lo"], core["y_lo"], core["x_hi"], core["y_hi"]],
            "candidate_frames": len(sel), "frames_integrated": 0,
            "support_frame_ids": [], "points_extracted": 0, "points_emitted": 0,
            "geometry_class": "unknown", "status": "unobservable", "confidence": None,
            "reason": "no_camera_support", "measurement_authoritative": False,
            "z_band": None, "z_source": "metric_depth_support",
            "local_sparse_points": plan["local_sparse_points"],
            "sparse_z_band": [v if np.isfinite(v) else None for v in (plan["z_lo"], plan["z_hi"])],
            "camera_support_method": "metric_depth_footprint",
        }
        lo, hi = np.inf, -np.inf
        for index in sel:
            loaded = load(frame_specs[index])
            if loaded is None:
                continue
            _, depth, ext = loaded
            for points in depth_world_chunks(depth, ext, fx, fy, cx, cy, depth_trunc):
                inside = ((points[:, 0] >= window["x_lo"]) & (points[:, 0] < window["x_hi"])
                          & (points[:, 1] >= window["y_lo"]) & (points[:, 1] < window["y_hi"]))
                z = points[inside, 2]
                if len(z):
                    lo, hi = min(lo, float(z.min())), max(hi, float(z.max()))
            del loaded, depth
        if np.isfinite(lo):
            event["z_band"] = [lo, hi]
            n_slabs = max(1, int(np.ceil((hi - lo) / max_vertical_slab_m)))
            event.update(status="running", reason=None, n_vertical_slabs=n_slabs, slabs_completed=0)
            if tile_observer:
                tile_observer(dict(event))
            here = set()
            origin = tile_origin_from_specs(frame_specs, sel)
            for slab in range(n_slabs):
                zl = lo + slab * max_vertical_slab_m
                zh = np.nextafter(hi, np.inf) if slab == n_slabs - 1 else zl + max_vertical_slab_m
                log.info("tile %s slab %d/%d z=%.2f..%.2f", event["tile_id"], slab + 1, n_slabs, zl, zh)
                volume = _make_volume(o3d, voxel_m, sdf_trunc)
                count = 0
                extracted = emitted = 0
                for index in sel:
                    loaded = load(frame_specs[index])
                    if loaded is None:
                        continue
                    color, depth, ext = loaded
                    clipped = _clip_depth_to_box(depth, fx, fy, cx, cy, ext,
                                                 window["x_lo"], window["x_hi"], window["y_lo"], window["y_hi"],
                                                 zl - sdf_trunc, zh + sdf_trunc)
                    clipped[~np.isfinite(clipped) | (clipped >= depth_trunc)] = 0
                    if np.any(clipped > 0):
                        _integrate(o3d, volume, intr, color, clipped,
                                   recenter_extrinsic(ext, origin), depth_trunc)
                        here.add(index)
                        used.add(index)
                        count += 1
                    del loaded, color, depth, clipped
                if count:
                    cloud = volume.extract_point_cloud()
                    points = np.asarray(cloud.points, dtype=np.float64).copy()
                    colors = np.asarray(cloud.colors)
                    if len(points):
                        # Open3D fused this slab in a local metric frame; restore mission CRS before
                        # XYZ core cropping and before handing points to the streaming sink.
                        points += origin
                    extracted = len(points)
                    event["points_extracted"] += extracted
                    keep = ((points[:, 0] >= core["x_lo"]) & (points[:, 0] < core["x_hi"])
                            & (points[:, 1] >= core["y_lo"]) & (points[:, 1] < core["y_hi"])
                            & (points[:, 2] >= zl - sdf_trunc if slab == 0 else points[:, 2] >= zl)
                            & (points[:, 2] < zh + sdf_trunc if slab == n_slabs - 1 else points[:, 2] < zh))
                    emitted = int(keep.sum())
                    event["points_emitted"] += emitted
                    if keep.any():
                        if sink:
                            sink(points[keep], colors[keep])
                        else:
                            all_points.append(points[keep])
                            all_colors.append(colors[keep])
                    del cloud, points, colors
                del volume
                gc.collect()
                log.info("tile %s slab %d/%d: frames=%d extracted=%d emitted=%d",
                         event["tile_id"], slab + 1, n_slabs, count, extracted, emitted)
                event["frames_integrated"] = len(here)
                event["support_frame_ids"] = [frame_specs[i].get("name", str(i)) for i in sorted(here)]
                event["slabs_completed"] = slab + 1
                if tile_observer:
                    tile_observer(dict(event))
            event["frames_integrated"] = len(here)
            event["support_frame_ids"] = [frame_specs[i].get("name", str(i)) for i in sorted(here)]
            event["n_vertical_slabs"] = n_slabs
            if event["points_emitted"]:
                event.update(status="reconstructed", geometry_class="reconstructed", reason=None)
            else:
                event["status"] = "unobservable"
                event["reason"] = ("no_surface_in_core" if event["points_extracted"] else
                                   "fusion_empty" if here else "no_valid_depth_support")
        elif sel:
            event["reason"] = "no_valid_depth_support"
        if tile_observer:
            tile_observer(dict(event))
    if not used:
        raise RuntimeError("TSDF fusion integrated 0 frames (no valid metric depth)")
    empty = np.empty((0, 3), float)
    return (np.concatenate(all_points) if all_points else empty,
            np.concatenate(all_colors) if all_colors else empty, len(used))
