"""S6 — Global bundle adjustment / consistency.

S2 already produces a locally-optimized, GPS-anchored spine. S6 is the *global* pass: when GLOMAP is
installed it runs a global refine; either way it computes global consistency metrics (per-frame GPS
residual, revisit/loop-closure candidates) and emits the poses S7 consumes. If GLOMAP is absent the
poses pass through unchanged and the stage reports that honestly (AGENTS.md §5).
"""
from __future__ import annotations

from typing import Any

from ..bundle import Bundle
from ..config import DrishtiConfig
from ..geo.align import rmse, umeyama
from ..geo.crs import haversine_m
from ..geo.project import unproject_to_lonlat
from ..runtime import ComputeNeed, Runtime
from .base import Stage, StageResult, register_stage


@register_stage
class GlobalStage(Stage):
    name = "s6_global"
    requires = ("s2_poses",)
    compute = ComputeNeed(cpu_only=True)

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"poses": {"use_glomap": cfg.poses.use_glomap}, "spine": cfg.spine.model_dump(), "contract_version": 3}

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        import numpy as np

        poses_doc = bundle.read_json("s2_poses/poses.json")
        sparse_doc = bundle.read_json("s2_poses/sparse_points.json")
        poses = poses_doc["poses"]
        if len(poses) < 2:
            raise ValueError("s6_global: need >=2 poses.")

        centers = np.asarray([p["center_crs"] for p in poses], dtype=float)

        # --- per-frame consistency: neighbor spacing (drift proxy) ---
        steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        median_step = float(np.median(steps)) if len(steps) else 0.0

        # --- revisit / loop-closure candidates: near in space, far in sequence ---
        n = len(poses)
        gap = max(5, int(0.05 * n))
        radius = 2.0 * cfg.spine.gnss_sigma_m  # from config, not hardcoded
        loops = 0
        lats = [p.get("lat") for p in poses]
        lons = [p.get("lon") for p in poses]
        for i in range(n):
            for j in range(i + gap, n):
                if None in (lats[i], lons[i], lats[j], lons[j]):
                    continue
                if haversine_m(lats[i], lons[i], lats[j], lons[j]) <= radius:
                    loops += 1
                    break

        refinement = {"status": "skipped", "requested": bool(cfg.poses.use_glomap)}
        refined_poses = poses
        if cfg.poses.use_glomap:
            try:
                from ..recon.glomap import run_glomap

                kf_doc = bundle.read_json("s1_frameqa/keyframes.json")
                image_dir = bundle.artifact_path(kf_doc["keyframes"][0]["path"]).parent
                colmap_dir = bundle.stage_dir("s2_poses") / "colmap"
                result = run_glomap(
                    colmap_dir / "database.db", image_dir, bundle.stage_dir(self.name) / "glomap"
                )
                by_name = {p.name: p for p in result.poses}
                old_by_name = {p["name"]: p for p in poses}
                common = [name for name in by_name if name in old_by_name]
                if len(common) < 3:
                    raise RuntimeError(f"only {len(common)} GLOMAP/S2 pose correspondences")
                refined_visual = np.asarray([by_name[name].center for name in common], dtype=float)
                target_local = np.asarray([old_by_name[name]["center_crs"] for name in common], dtype=float)
                # GLOMAP has its own arbitrary frame. Align it back to S2's georeferenced frame before
                # replacing poses; otherwise a successful global run would silently lose CRS correctness.
                fit = umeyama(refined_visual, target_local, with_scale=True)
                # Retain the metric frame/calibration contract for dense fusion. Accepting new poses
                # while keeping old sparse points or lens parameters would corrupt S4 metric anchors.
                if result.intrinsics != poses_doc["intrinsics"]:
                    raise ValueError("GLOMAP changed camera calibration; reject until images can be rectified consistently")
                refined_poses = []
                if bundle.manifest.crs is None or bundle.manifest.crs.epsg is None:
                    raise RuntimeError("s6_global: manifest CRS is missing")
                epsg = bundle.manifest.crs.epsg
                for old in poses:
                    candidate = by_name.get(old["name"])
                    if candidate is None:
                        refined_poses.append(old)
                        continue
                    center = fit.apply(candidate.center.reshape(1, 3)).reshape(3)
                    R = candidate.rotation @ np.asarray(fit.R, dtype=float).T
                    lon, lat = unproject_to_lonlat([center[0]], [center[1]], epsg)
                    refined_poses.append({
                        **old, "center_crs": center.tolist(), "R": R.tolist(),
                        "lon": lon[0], "lat": lat[0], "alt": float(center[2]),
                    })
                refinement = {
                    "status": "optimized", "model_path": bundle.relpath(result.model_path),
                    "n_registered": len(result.poses), "n_common": len(common),
                    "alignment_rmse_m": round(float(rmse(fit.apply(refined_visual), target_local)), 4),
                    "sim3_scale": float(fit.scale),
                }
                sparse_doc = {"crs_epsg": epsg, "n_points": len(result.sparse_points),
                              "points": fit.apply(result.sparse_points).tolist()}
            except Exception as exc:
                refinement = {"status": "failed", "error": str(exc)}

        from ..recon.camera import rectify_views
        views, pinhole = rectify_views(bundle, bundle.read_json("s1_frameqa/keyframes.json")["keyframes"], poses_doc["intrinsics"])
        views_ref = bundle.write_json(f"{self.name}/views.json", {"keyframes": views})
        refined_doc = {**poses_doc, "poses": refined_poses, "intrinsics": pinhole,
                       "source_intrinsics": poses_doc["intrinsics"], "views": views_ref,
                       "source": "glomap_refined" if refinement["status"] == "optimized" else "s2_passthrough"}
        poses_ref = bundle.write_json(f"{self.name}/poses_global.json", refined_doc)
        sparse_ref = bundle.write_json(f"{self.name}/sparse_points.json", sparse_doc)
        global_ref = bundle.write_json(f"{self.name}/global.json", {
            "n_poses": n,
            "median_step_m": round(median_step, 3),
            "loop_candidates": loops,
            "revisit_radius_m": round(radius, 3),
            "glomap": refinement,
        })

        degraded = None
        if refinement["status"] == "skipped":
            degraded = "GLOMAP global refinement disabled; poses passed through from S2"
        elif refinement["status"] == "failed":
            degraded = f"GLOMAP global refinement failed; poses passed through from S2: {refinement['error']}"
        conf = None  # a successful tool invocation is not a measured accuracy probability
        outputs = {"poses_global": poses_ref, "global": global_ref, "sparse_points": sparse_ref,
                   "views": views_ref, "rectified_images": bundle.relpath(bundle.stage_dir(self.name) / "rectified") if bundle.exists("s6_global/rectified") else views_ref}
        if views and views[0].get("valid_pixels"):
            outputs["valid_pixels"] = views[0]["valid_pixels"]
        return StageResult(
            outputs=outputs,
            metrics={"n_poses": n, "median_step_m": round(median_step, 3),
                "loop_candidates": loops, "glomap_status": refinement["status"],
                "glomap_alignment_rmse_m": refinement.get("alignment_rmse_m")},
            confidence_summary=conf,
            degraded_reason=degraded,
        )
