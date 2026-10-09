"""S7 — Dense reconstruction (TSDF fusion).

Fuses per-keyframe metric depth + color (with dynamic pixels already zeroed by S3/S4) into a scalable
TSDF volume and extracts a dense colored point cloud in the ground CRS. TSDF is the shipped dense method;
the Gaussian-splatting path requires gsplat and is selected via dense.method=gaussian.
"""
from __future__ import annotations

from typing import Any

from ..bundle import Bundle
from ..config import DrishtiConfig
from ..runtime import ComputeNeed, Runtime
from .base import Stage, StageResult, register_stage


@register_stage
class DenseStage(Stage):
    name = "s7_dense"
    requires = ("s4_depth", "s6_global")
    compute = ComputeNeed(vram_gb=6.0, openvino_ok=False)

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"dense": cfg.dense.model_dump(), "contract_version": 3}

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        import numpy as np

        from ..io.pointcloud import StreamingPlyWriter, write_ply_points
        from ..io.video import read_image
        from ..recon.depthfit import intrinsics_matrix

        if cfg.dense.method not in {"tsdf", "gaussian"}:
            raise RuntimeError(
                f"s7_dense: method '{cfg.dense.method}' not available in this build; "
                "available methods are tsdf and gaussian."
            )

        from ..logging import get_logger
        from ..recon.tsdf import depth_world_chunks, fuse_tsdf, fuse_tsdf_tiled, robust_z_band

        log = get_logger("drishti.s7_dense")

        poses_doc = bundle.read_json("s6_global/poses_global.json")
        depth_doc = bundle.read_json("s4_depth/depth.json")
        depth_by_index = {d["index"]: d for d in depth_doc["frames"]}

        intr = poses_doc["intrinsics"]
        K = intrinsics_matrix(intr.get("model", ""), intr.get("params", []))
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        width, height = int(intr["width"]), int(intr["height"])

        from pathlib import Path as _P
        kf_doc = bundle.read_json("s6_global/views.json")
        idx_by_name = {_P(k["path"]).name: k["index"] for k in kf_doc["keyframes"]}
        path_by_name = {_P(k["path"]).name: k["path"] for k in kf_doc["keyframes"]}

        # Re-iterable per-frame descriptors: camera center in the ground CRS (for spatial tiling) plus
        # the artifact paths + extrinsic needed to lazily materialize each frame during fusion.
        specs: list[dict] = []
        for p in poses_doc["poses"]:
            name = p["name"]
            midx = idx_by_name.get(name)
            d = depth_by_index.get(midx)
            if d is None or name not in path_by_name or not d.get("metric", False):
                continue
            R_wc = np.asarray(p["R"], dtype=float)
            center = np.asarray(p["center_crs"], dtype=float)
            extr = np.eye(4)
            extr[:3, :3] = R_wc
            extr[:3, 3] = -R_wc @ center
            specs.append({
                "name": name,
                "center_xy": (float(center[0]), float(center[1])),
                "depth": d["depth"], "color": path_by_name[name], "extr": extr,
            })

        if not specs:
            raise RuntimeError("s7_dense: no metric depth frames matched global poses and views")

        def _load(spec: dict):
            depth = np.load(bundle.artifact_path(spec["depth"]))
            color = read_image(bundle.artifact_path(spec["color"]))
            return color, depth, spec["extr"]

        if cfg.dense.tile.enabled:
            for spec in specs:
                mins, maxs = np.full(3, np.inf), np.full(3, -np.inf)
                depth = np.load(bundle.artifact_path(spec["depth"]), mmap_mode="r")
                for points in depth_world_chunks(depth, spec["extr"], fx, fy, cx, cy, cfg.dense.depth_trunc_m):
                    mins = np.minimum(mins, points.min(axis=0))
                    maxs = np.maximum(maxs, points.max(axis=0))
                if np.isfinite(mins).all():
                    # Tile camera membership is based on observed rays, not where the drone was overhead.
                    spec["footprint_xy"] = [mins[0], mins[1], np.nextafter(maxs[0], np.inf), np.nextafter(maxs[1], np.inf)]

        if cfg.dense.method == "gaussian":
            from ..io.pointcloud import write_ply_points
            from ..recon.gaussian import train_metric_gaussians

            result = train_metric_gaussians(
                specs, _load, fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height,
                voxel_m=cfg.dense.tsdf_voxel_m, iters=cfg.dense.gaussian.iters,
                max_gaussians=cfg.dense.gaussian.max_gaussians,
                view_batch=cfg.dense.gaussian.view_batch,
                checkpoint_every=cfg.dense.gaussian.checkpoint_every,
                output_dir=bundle.stage_dir(self.name), seed=cfg.run.seed,
            )
            ply = bundle.stage_dir(self.name) / "gaussians.ply"
            points_crs = result.points + result.origin_crs
            write_ply_points(ply, points_crs, result.colors, coordinate_dtype="float64")
            dense_ref = bundle.write_json(f"{self.name}/dense.json", {
                "method": "gaussian", "voxel_m": cfg.dense.tsdf_voxel_m,
                "n_frames_fused": len(specs), "n_points": result.n_gaussians,
                "cloud": bundle.relpath(ply), "bbox_min": points_crs.min(axis=0).tolist(),
                "bbox_max": points_crs.max(axis=0).tolist(),
                "origin_crs": result.origin_crs.tolist(),
                "geometry_class": "reconstructed", "measurement_authoritative": False,
                "gaussian_checkpoint": bundle.relpath(result.checkpoint),
                "gaussian_iterations": result.iterations,
                "final_rgb_loss": result.final_rgb_loss,
                "final_depth_loss": result.final_depth_loss,
                "geometry_classes": ["observed", "reconstructed", "inferred", "unknown"],
            })
            return StageResult(
                outputs={"dense": dense_ref, "cloud": bundle.relpath(ply),
                         "gaussians": bundle.relpath(result.checkpoint)},
                metrics={"n_points": result.n_gaussians, "n_frames_fused": len(specs),
                         "method": "gaussian", "iterations": result.iterations,
                         "final_rgb_loss": result.final_rgb_loss,
                         "final_depth_loss": result.final_depth_loss,
                         "measurement_authoritative": False},
                confidence_summary=None,
            )

        voxel = cfg.dense.tsdf_voxel_m
        tile = cfg.dense.tile
        ply = bundle.stage_dir(self.name) / "dense.ply"
        tile_states: dict[str, dict] = {}
        sparse_points = None
        if tile.enabled:
            # Vertical surface envelope from S2's metric sparse points (COLMAP-triangulated true surface,
            # same ground CRS as the poses). Clipping each depth map to this Z band before fusion drops the
            # gross far-field flyers that noisy monocular depth scatters across a thick vertical slab — the
            # real per-tile RAM driver — while keeping all depth the sparse geometry actually supports.
            # Fail-soft: if sparse points are missing/degenerate the band is open (no vertical clip).
            z_lo, z_hi = -float("inf"), float("inf")
            try:
                sparse = np.asarray(
                    bundle.read_json("s6_global/sparse_points.json")["points"], dtype=float
                )
                if sparse.ndim == 2 and sparse.shape[0] >= 2 and sparse.shape[1] >= 3:
                    sparse_points = sparse[:, :3]
                    z_lo, z_hi = robust_z_band(sparse[:, 2], margin_m=cfg.dense.ground_band_margin_m)
            except (FileNotFoundError, KeyError, ValueError) as e:
                log.warning("s7_dense: no usable sparse points for Z band (%s); vertical clip disabled", e)
            if np.isfinite(z_lo) and np.isfinite(z_hi):
                log.info("s7_dense: ground Z band %.1f..%.1f m (%.1f m, margin=%.1f m) from %d sparse pts",
                         z_lo, z_hi, z_hi - z_lo, cfg.dense.ground_band_margin_m, len(sparse))
            else:
                log.warning("s7_dense: vertical Z clip disabled (open band); per-tile RAM bounded by tile_m only")

            # Memory-bounded: fuse one ground tile at a time so peak RAM tracks tile extent, not the whole
            # aerial scene, AND stream each tile's points straight to the PLY on disk (freeing them) so RAM
            # never holds the full cloud — the ~100M-point accumulation that OOM-killed the whole-scene run.
            def _observe_tile(event: dict) -> None:
                tile_states[event["tile_id"]] = event

            with StreamingPlyWriter(ply, with_color=True, coordinate_dtype="float64") as writer:
                _, _, used = fuse_tsdf_tiled(
                    specs, _load, fx, fy, cx, cy, width, height,
                    voxel_m=voxel, sdf_trunc=4.0 * voxel, depth_trunc=cfg.dense.depth_trunc_m,
                    tile_m=tile.tile_m, overlap_m=tile.overlap_m, z_lo=z_lo, z_hi=z_hi,
                    sink=writer.add, surface_points=sparse_points,
                    z_margin_m=cfg.dense.ground_band_margin_m,
                    min_local_surface_points=cfg.dense.min_local_surface_points,
                    tile_observer=_observe_tile,
                    max_vertical_slab_m=cfg.dense.max_vertical_slab_m,
                )
            n_points = writer.count
            if n_points == 0:
                raise RuntimeError("s7_dense: TSDF produced an empty cloud.")
            if writer.bbox_min is None or writer.bbox_max is None:
                raise RuntimeError("s7_dense: streaming writer has no bounding box")
            bbox_min = writer.bbox_min.tolist()
            bbox_max = writer.bbox_max.tolist()
        else:
            points, colors, used = fuse_tsdf(
                (_load(s) for s in specs), fx, fy, cx, cy, width, height,
                voxel_m=voxel, sdf_trunc=4.0 * voxel, depth_trunc=cfg.dense.depth_trunc_m,
            )
            if len(points) == 0:
                raise RuntimeError("s7_dense: TSDF produced an empty cloud.")
            write_ply_points(ply, points, colors, coordinate_dtype="float64")
            n_points = int(len(points))
            bbox_min = points.min(axis=0).tolist()
            bbox_max = points.max(axis=0).tolist()
            tile_states["scene"] = {
                "tile_id": "scene", "geometry_class": "reconstructed", "status": "reconstructed",
                "reason": None, "candidate_frames": len(specs), "frames_integrated": used,
                "points_extracted": n_points, "points_emitted": n_points,
                "confidence": round(used / max(1, len(specs)), 4),
            }

        tile_states_ref = bundle.write_json(f"{self.name}/tile_states.json", {
            "geometry_classes": ["observed", "reconstructed", "inferred", "unknown"],
            "authoritative_classes": ["observed", "validated_reconstructed"],
            "tiles": list(tile_states.values()),
        })
        reconstructed_tiles = sum(
            1 for state in tile_states.values() if state.get("geometry_class") == "reconstructed"
        )
        unknown_tiles = sum(
            1 for state in tile_states.values() if state.get("geometry_class") == "unknown"
        )

        metric_aligned = int(depth_doc.get("n_metric_aligned", 0))
        depth_frames = int(depth_doc.get("n_frames", 0))
        dense_ref = bundle.write_json(f"{self.name}/dense.json", {
            "method": "tsdf", "voxel_m": voxel, "n_frames_fused": used,
            "n_points": n_points, "cloud": bundle.relpath(ply),
            "bbox_min": bbox_min, "bbox_max": bbox_max,
            "geometry_class": "reconstructed",
            "measurement_authoritative": False,
            "geometry_classes": ["observed", "reconstructed", "inferred", "unknown"],
            "tile_states": tile_states_ref,
            "n_reconstructed_tiles": reconstructed_tiles,
            "n_unknown_tiles": unknown_tiles,
            "metric_aligned_frame_fraction": round(metric_aligned / max(1, depth_frames), 4),
        })

        return StageResult(
            outputs={"dense": dense_ref, "cloud": bundle.relpath(ply)},
            metrics={"method": "tsdf", "n_points": n_points, "n_frames_fused": used,
                     "voxel_m": voxel, "n_reconstructed_tiles": reconstructed_tiles,
                     "n_unknown_tiles": unknown_tiles,
                     "metric_aligned_frame_fraction": round(metric_aligned / max(1, depth_frames), 4),
                     "measurement_authoritative": False},
            confidence_summary=None,
            confidence_ref=tile_states_ref,
            degraded_reason=f"{unknown_tiles} tiles have no reconstructed surface; inspect tile_states.json" if unknown_tiles else None,
        )
