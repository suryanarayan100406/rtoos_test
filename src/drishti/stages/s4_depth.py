"""S4 — bounded learned geometry inference and evidence-anchored metric camera-Z depth."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..bundle import Bundle
from ..config import DrishtiConfig
from ..runtime import ComputeNeed, Runtime
from .base import Stage, StageResult, register_stage


@register_stage
class DepthStage(Stage):
    name = "s4_depth"
    requires = ("s2_poses", "s3_masking", "s6_global")
    compute = ComputeNeed(vram_gb=4.0, openvino_ok=False)

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"depth": cfg.depth.model_dump(), "models": cfg.models.model_dump(), "contract_version": 3}

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        import numpy as np

        from ..io.video import read_image
        from ..models.depth import DepthAnythingV2
        from ..models.geometry import DepthAnything3Backend, MapAnythingBackend, remap_prediction
        from ..recon.depthfit import align_depth, intrinsics_matrix, project_points
        from ..runtime.deps import require

        cv2 = require("cv2", purpose="depth raster I/O")
        poses_doc = bundle.read_json("s6_global/poses_global.json")
        sparse = np.asarray(bundle.read_json("s6_global/sparse_points.json")["points"], dtype=float).reshape(-1, 3)
        mask_by_index = {m["index"]: m["mask"] for m in bundle.read_json("s3_masking/masks.json")["masks"]}
        frames = bundle.read_json("s6_global/views.json")["keyframes"]
        by_name = {Path(k["path"]).name: k for k in frames}
        poses = sorted((p for p in poses_doc["poses"] if p["name"] in by_name),
                       key=lambda p: by_name[p["name"]]["t"])
        intr = poses_doc["intrinsics"]
        K = intrinsics_matrix(intr["model"], intr["params"])
        learned: Any = None
        relative: Any = None
        if cfg.depth.model.startswith("da3_"):
            learned = DepthAnything3Backend(rt, cfg.depth.model)
        elif cfg.depth.model == "mapanything_apache":
            learned = MapAnythingBackend(rt, cfg.depth.model)
        else:
            relative = DepthAnythingV2(rt, cfg.depth.model)
        # An enabled secondary geometry backend must actually run, never be ignored silently.
        secondary = None
        if cfg.depth.feedforward_geometry.enabled:
            model = cfg.depth.feedforward_geometry.model
            secondary = (MapAnythingBackend(rt, model) if model == "mapanything_apache"
                         else DepthAnything3Backend(rt, model))

        depth_dir = bundle.stage_dir(self.name) / "depth"
        depth_dir.mkdir(exist_ok=True)
        index = []
        window = max(1, cfg.depth.geometry_window)
        if secondary:
            window = min(window, max(1, cfg.depth.feedforward_geometry.window))
        for start in range(0, len(poses), window):
            batch = poses[start:start + window]
            paths = [bundle.artifact_path(by_name[p["name"]]["path"]) for p in batch]
            # Subtract the CRS origin in float64 BEFORE creating neural float32 extrinsics.
            origin = np.asarray(batch[0]["center_crs"], dtype=np.float64)
            extrinsics = []
            for pose in batch:
                R = np.asarray(pose["R"], dtype=np.float64)
                ext = np.eye(4)
                ext[:3, :3] = R
                ext[:3, 3] = -R @ (np.asarray(pose["center_crs"], dtype=np.float64) - origin)
                extrinsics.append(ext)
            Ks = np.repeat(K[None], len(batch), axis=0)
            predictions = learned.predict(paths, np.asarray(extrinsics), Ks) if learned else None
            secondary_predictions = secondary.predict(paths, np.asarray(extrinsics), Ks) if secondary else None
            for i, pose in enumerate(batch):
                frame = by_name[pose["name"]]
                img = read_image(paths[i])
                h, w = img.shape[:2]
                if (h, w) != (intr["height"], intr["width"]):
                    raise ValueError(f"frame {frame['index']} size differs from global camera calibration")
                if predictions is not None:
                    pred = predictions[i]
                    raw, model_conf = remap_prediction(pred, K, h, w)
                    native_metric = pred.metric
                else:
                    raw, model_conf, native_metric = relative.predict(img), None, False
                excluded = np.zeros((h, w), dtype=bool)
                for path, dynamic in ((mask_by_index.get(frame["index"]), True),
                                      (frame.get("valid_pixels"), False)):
                    if path:
                        mask = cv2.imread(str(bundle.artifact_path(path)), cv2.IMREAD_GRAYSCALE)
                        if mask is None or mask.shape != (h, w):
                            raise ValueError(f"invalid pixel mask for frame {frame['index']}: {path}")
                        excluded |= mask > 0 if dynamic else mask == 0
                # Mask before fitting: a sparse point on a dynamic object/padded pixel cannot anchor scale.
                raw = raw.copy()
                raw[excluded] = 0.0
                uv, z = project_points(K, np.asarray(pose["R"]), np.asarray(pose["center_crs"]), sparse)
                degraded = None
                alignment = None
                try:
                    if cfg.depth.scale_align == "sparse":
                        metric, alignment = align_depth(raw, uv, z, inverse=relative is not None,
                                                        min_anchors=cfg.depth.min_metric_anchors)
                    elif native_metric:
                        metric = raw.astype(np.float32)
                    else:
                        raise ValueError("relative depth requires metric anchoring")
                except ValueError as exc:
                    if native_metric:
                        metric = raw.astype(np.float32)
                        degraded = f"native model metric scale, sparse validation unavailable: {exc}"
                    else:
                        # Preserve relative depth separately for diagnostics; never integrate it in metres.
                        np.save(depth_dir / f"{frame['index']}.relative.npy", raw)
                        metric = np.zeros_like(raw, dtype=np.float32)
                        degraded = f"not metric; excluded from fusion: {exc}"
                disagreement = None
                secondary_degraded = None
                if secondary_predictions is not None:
                    alt, _ = remap_prediction(secondary_predictions[i], K, h, w)
                    alt[excluded] = 0.0
                    if not secondary_predictions[i].metric:
                        try:
                            alt, _ = align_depth(alt, uv, z, inverse=False,
                                                 min_anchors=cfg.depth.min_metric_anchors)
                        except ValueError as exc:
                            secondary_degraded = f"non-metric secondary depth; comparison unavailable: {exc}"
                    if secondary_degraded is None:
                        common = (metric > 0) & (alt > 0) & np.isfinite(metric) & np.isfinite(alt)
                        if common.any():
                            disagreement = np.zeros_like(metric)
                            disagreement[common] = np.abs(metric[common] - alt[common]) / np.maximum(metric[common], alt[common])
                            # Compare metric quantities only; relative neural depth is not metres.
                            np.save(depth_dir / f"{frame['index']}.disagreement.npy", disagreement)
                metric[~np.isfinite(metric) | (metric <= 0)] = 0.0
                metric[excluded] = 0.0
                dpath = depth_dir / f"{frame['index']}.npy"
                np.save(dpath, metric)
                conf_path = None
                if model_conf is not None:
                    model_conf[metric <= 0] = 0.0
                    conf_path = depth_dir / f"{frame['index']}.confidence.npy"
                    np.save(conf_path, model_conf)
                index.append({
                    "name": pose["name"], "index": frame["index"], "depth": bundle.relpath(dpath),
                    "units": "m", "depth_convention": "camera_z", "metric": bool(np.any(metric > 0)),
                    "validated_against_sparse": alignment is not None,
                    "n_anchors": 0 if alignment is None else alignment["n_anchors"],
                    "resid_m": None if alignment is None else alignment["resid_m"],
                    "alignment": alignment, "source": cfg.depth.model, "degraded": degraded,
                    "confidence_map": None if conf_path is None else bundle.relpath(conf_path),
                    "confidence_semantics": "uncalibrated_model_score" if conf_path else None,
                    "disagreement_median_relative": float(np.median(disagreement[common])) if disagreement is not None else None,
                    "secondary_degraded": secondary_degraded,
                })
            del predictions, secondary_predictions
        n_metric = sum(f["metric"] for f in index)
        if not n_metric:
            raise RuntimeError("S4 produced zero metric depth frames; cannot reconstruct a metric scene")
        residuals = [f["resid_m"] for f in index if f["resid_m"] is not None]
        spec = learned.spec if learned else relative.spec
        depth_ref = bundle.write_json(f"{self.name}/depth.json", {
            "contract_version": 3, "model": cfg.depth.model, "model_spec": asdict(spec),
            "n_frames": len(index), "n_metric_aligned": n_metric, "frames": index,
            "units": "m", "depth_convention": "camera_z",
        })
        issues = [f for f in index if f["degraded"] or f["secondary_degraded"]]
        return StageResult(
            outputs={"depth": depth_ref, "depth_dir": bundle.relpath(depth_dir)},
            metrics={"model": cfg.depth.model, "n_frames": len(index), "n_metric_aligned": n_metric,
                     "frac_metric_aligned": n_metric / len(index),
                     "median_resid_m": float(np.median(residuals)) if residuals else None,
                     "secondary_geometry_model": secondary.spec.name if secondary else None},
            confidence_summary=None,  # aligned-frame coverage is not a calibrated confidence probability
            degraded_reason=f"{len(issues)} frames have metric/depth evidence limitations" if issues else None,
        )
