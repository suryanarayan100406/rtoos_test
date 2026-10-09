"""S2 — Camera poses & metric spine.

COLMAP recovers intrinsics + poses in an arbitrary visual frame; we then anchor that frame to the
ground with a Sim(3) (scale+rotation+translation) fit to the GPS track in the derived projected CRS.
The fit residual (RMSE) is a real, reported accuracy proxy. If ``gtsam`` and inertial data are present,
a factor-graph tightening refines the spine; otherwise that step is skipped and reported as such — the
poses are still metric and georeferenced (AGENTS.md §5, honesty policy).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..bundle import Bundle
from ..config import DrishtiConfig
from ..geo.align import rmse, umeyama
from ..geo.project import project_lonlat, unproject_to_lonlat
from ..runtime import ComputeNeed, Runtime
from ..runtime.deps import available
from .base import Stage, StageResult, register_stage


@register_stage
class PosesStage(Stage):
    name = "s2_poses"
    requires = ("s1_frameqa",)
    # SfM is CPU-bound in COLMAP; matching can use CUDA on cloud but works on CPU locally.
    compute = ComputeNeed(cpu_only=True)

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"poses": cfg.poses.model_dump(), "spine": cfg.spine.model_dump(), "contract_version": 2}

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        import numpy as np

        from ..recon.sfm import run_colmap_sfm

        kf_doc = bundle.read_json("s1_frameqa/keyframes.json")
        keyframes = kf_doc["keyframes"]
        if len(keyframes) < 3:
            raise ValueError(
                f"s2_poses: need >=3 keyframes to reconstruct, got {len(keyframes)}."
            )

        image_paths = [Path(bundle.artifact_path(k["path"])) for k in keyframes]
        gps_by_name = {Path(k["path"]).name: k for k in keyframes}

        # --- Structure-from-Motion (real COLMAP) ---
        sfm = run_colmap_sfm(
            image_paths, work_dir=bundle.stage_dir(self.name) / "colmap",
            matcher=cfg.poses.matcher, self_calibrate=cfg.poses.self_calibrate_intrinsics,
        )
        if sfm.n_registered < 3:
            raise RuntimeError(
                f"s2_poses: COLMAP registered only {sfm.n_registered} images; "
                "cannot form a metric spine."
            )

        # --- correspondences: visual centers <-> GPS in projected CRS ---
        if bundle.manifest.crs is None or bundle.manifest.crs.epsg is None:
            raise RuntimeError("s2_poses: manifest CRS is missing")
        epsg = bundle.manifest.crs.epsg
        if epsg is None:
            raise ValueError("s2_poses: no CRS in manifest (s0_ingest must run first).")

        src, dst_ll, matched_pose_indices = [], [], []
        pose_index_by_name = {p.name: i for i, p in enumerate(sfm.poses)}
        for p in sfm.poses:
            g = gps_by_name.get(p.name)
            if g and g.get("lat") is not None and g.get("lon") is not None:
                src.append(p.center)
                dst_ll.append((g["lon"], g["lat"], g.get("alt") or 0.0))
                matched_pose_indices.append(pose_index_by_name[p.name])
        if len(src) < 3:
            raise RuntimeError(
                f"s2_poses: only {len(src)} registered frames have GPS; need >=3 to georeference."
            )

        lons = [d[0] for d in dst_ll]
        lats = [d[1] for d in dst_ll]
        alts = [d[2] for d in dst_ll]
        east, north = project_lonlat(lons, lats, epsg)
        origin = (float(np.mean(east)), float(np.mean(north)), float(np.mean(alts)))
        src_arr = np.asarray(src, dtype=float)
        dst_arr = np.column_stack([np.asarray(east) - origin[0],
                                   np.asarray(north) - origin[1],
                                   np.asarray(alts) - origin[2]])

        # --- Sim(3) alignment: visual frame -> ground (metric spine core) ---
        sim = umeyama(src_arr, dst_arr, with_scale=True)
        fit = sim.apply(src_arr)
        fit_rmse = rmse(fit, dst_arr)

        # Optional, real factor-graph pass over visual relative motion + GNSS position factors. A module
        # probe alone is not enough: the graph is considered active only after optimize() returns and its
        # post-fit residual is recorded. IMU is reported separately until a dataset-specific binary IMU
        # adapter supplies calibrated measurements and camera-to-IMU extrinsics.
        all_src = np.asarray([p.center for p in sfm.poses], dtype=float)
        optimized_local = sim.apply(all_src)
        factor_graph: dict[str, Any] = {"status": "skipped"}
        imu = bundle.manifest.dataset_snapshot.get("optional", {}).get("imu") if bundle.manifest.dataset_snapshot else None
        if available("gtsam") and cfg.spine.use_imu != "off":
            try:
                from ..recon.spine import optimize_position_spine

                proposed_local, factor_graph = optimize_position_spine(
                    optimized_local,
                    {i: dst_arr[j] for j, i in enumerate(matched_pose_indices)},
                    horizontal_sigma_m=cfg.spine.gnss_sigma_m,
                    vertical_sigma_m=cfg.spine.gnss_sigma_v_m,
                )
                # A position-only graph cannot deform cameras without retriangulating every landmark.
                # Fit a coherent Sim(3) to the optimized proposal and transform poses + sparse points
                # together. Record both the proposal and the accepted coherent residual.
                proposed_sim = umeyama(all_src, proposed_local, with_scale=True)
                proposed_rmse = rmse(proposed_sim.apply(src_arr), dst_arr)
                factor_graph["coherent_candidate_rmse_m"] = proposed_rmse
                factor_graph["accepted"] = proposed_rmse <= fit_rmse
                if proposed_rmse <= fit_rmse:
                    sim = proposed_sim
                    optimized_local = sim.apply(all_src)
                    fit_rmse = proposed_rmse
                factor_graph["status"] = "optimized"
            except Exception as exc:
                factor_graph = {"status": "failed", "error": str(exc)}

        R_a = np.asarray(sim.R, dtype=float)

        # --- apply to every registered pose ---
        out_poses: list[dict[str, Any]] = []
        for pose_index, p in enumerate(sfm.poses):
            c_geo = optimized_local[pose_index].reshape(3) + np.asarray(origin)
            R_cam = np.asarray(p.R, dtype=float)
            R_cam_geo = R_cam @ R_a.T  # orientation in the ground frame
            lon, lat = unproject_to_lonlat([c_geo[0]], [c_geo[1]], epsg)
            out_poses.append({
                "name": p.name,
                "center_crs": [float(x) for x in c_geo],   # easting, northing, alt (m)
                "R": R_cam_geo.tolist(),
                "lat": lat[0], "lon": lon[0], "alt": float(c_geo[2]),
            })

        # --- sparse points into the ground frame ---
        pts = np.asarray(sfm.points_xyz, dtype=float) if sfm.points_xyz else np.zeros((0, 3))
        pts_geo = (sim.apply(pts) + np.asarray(origin)) if len(pts) else pts

        poses_ref = bundle.write_json(f"{self.name}/poses.json", {
            "crs_epsg": epsg, "intrinsics": sfm.intrinsics,
            "n_registered": sfm.n_registered, "n_input": sfm.n_input_images,
            "poses": out_poses,
        })
        spine_ref = bundle.write_json(f"{self.name}/spine.json", {
            "align": "umeyama_sim3",
            "scale": sim.scale, "R": R_a.tolist(), "t": list(map(float, sim.t)),
            "origin_crs": list(origin), "epsg": epsg,
            "fit_rmse_m": round(float(fit_rmse), 3),
            "n_correspondences": len(src),
            "factor_graph": factor_graph,
        })
        bundle.write_json(f"{self.name}/sparse_points.json", {
            "crs_epsg": epsg, "n_points": int(len(pts_geo)),
            "points": pts_geo.tolist(),
        })

        degraded = None
        if factor_graph.get("status") == "skipped":
            degraded = "GTSAM position factor graph skipped (gtsam unavailable or disabled); spine from Sim(3) GPS fit"
        elif factor_graph.get("status") == "failed":
            degraded = f"GTSAM position factor graph failed; spine from Sim(3) GPS fit: {factor_graph.get('error')}"
        elif imu:
            degraded = "GTSAM optimized visual+GNSS positions; IMU input present but no calibrated IMU factor adapter ran"

        # confidence: shrinks as fit RMSE grows relative to GNSS sigma
        conf = max(0.0, min(1.0, cfg.spine.gnss_sigma_m / (fit_rmse + 1e-6)))
        return StageResult(
            outputs={"poses": poses_ref, "spine": spine_ref,
                     "sparse_points": f"{self.name}/sparse_points.json"},
            metrics={
                "n_registered": sfm.n_registered,
                "n_input_keyframes": len(keyframes),
                "registration_rate": round(sfm.n_registered / max(1, len(keyframes)), 3),
                "spine_fit_rmse_m": round(float(fit_rmse), 3),
                "sim3_scale": round(float(sim.scale), 6),
                "n_sparse_points": int(len(pts_geo)),
                "factor_graph_status": factor_graph.get("status"),
                "factor_graph_postfit_gnss_rmse_m": factor_graph.get("postfit_gnss_rmse_m"),
            },
            confidence_summary=round(conf, 3),
            degraded_reason=degraded,
        )
