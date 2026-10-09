"""S3 — Dynamic-object masking.

Detects dynamic classes (people, vehicles, …) with RT-DETR and writes per-keyframe binary masks so
moving objects are excluded from the static 3D reconstruction. Masks are real detector output grown by
a morphological margin; SAM2 pixel-refinement is an optional enhancement (enabled when installed +
configured). Value 255 = dynamic/exclude, 0 = static/keep.
"""
from __future__ import annotations

from typing import Any

from ..bundle import Bundle
from ..config import DrishtiConfig
from ..runtime import ComputeNeed, Runtime
from .base import Stage, StageResult, register_stage

# Expand coarse class names in config to the concrete COCO labels RT-DETR emits.
_CLASS_EXPANSION = {
    "animal": {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"},
    "vehicle": {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane"},
}


def _wanted_labels(dynamic_classes: list[str]) -> set[str]:
    want: set[str] = set()
    for c in dynamic_classes:
        want.add(c)
        want |= _CLASS_EXPANSION.get(c, set())
    return want


def _capability_metadata(cfg: DrishtiConfig, decision) -> dict[str, Any]:
    """Describe requested versus executed perception methods without implying absent inference ran."""
    requested = {
        "detector": cfg.masking.detector,
        "segmenter": cfg.masking.segmenter,
        "tracker": cfg.masking.tracker,
        "motion": cfg.masking.motion,
        "semantic_classes": list(cfg.masking.semantic_classes),
    }
    executed = {
        "detector": cfg.masking.detector,
        "box_to_mask": True,
        "segmenter": None,
        "tracker": None,
        "motion": None,
        "semantic_classes": [],
    }
    not_executed = [name for name in ("segmenter", "tracker", "motion") if requested[name]]
    if requested["semantic_classes"]:
        not_executed.append("semantic_classes")
    return {
        "requested": requested,
        "executed": executed,
        "not_executed": not_executed,
        "tier": decision.tier.name,
        "tier_level": decision.tier.level,
        "tier_considered": decision.considered,
    }


@register_stage
class MaskingStage(Stage):
    name = "s3_masking"
    requires = ("s2_poses", "s6_global")
    # RT-DETR runs on the T4 when available, else CPU/OpenVINO locally.
    compute = ComputeNeed(vram_gb=3.0, openvino_ok=True)

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"masking": cfg.masking.model_dump(), "contract_version": 3}

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        import numpy as np

        from ..io.video import read_image
        from ..models.detect import RTDetrDetector
        from ..runtime.deps import require
        from ..runtime.reliability import choose, tier

        cv2 = require("cv2", purpose="mask raster I/O")
        kf_doc = bundle.read_json("s6_global/views.json")
        keyframes = kf_doc["keyframes"]
        if not keyframes:
            raise ValueError("s3_masking: no keyframes to mask.")

        # Honest degradation decision: SAM2 pixel-refinement is the preferred (L0) method but is not
        # implemented in this pipeline version, so we run the detector-box path (L2). The ladder makes
        # the fallback reason + confidence ceiling consistent with every other stage. When a SAM2
        # refinement branch is added, flip this tier's availability to ``available("sam2")``.
        decision = choose([
            tier(0, "sam2_refined", available=False,
                 note="SAM2 pixel refinement is not built in this pipeline version"),
            tier(2, "rtdetr_boxes", available=True,
                 note="RT-DETR detection boxes grown by a morphological margin"),
        ])

        detector = RTDetrDetector(rt, cfg.masking.detector)
        want = _wanted_labels(cfg.masking.dynamic_classes)
        dilate = int(cfg.masking.dynamic_dilate_px)
        kernel = np.ones((dilate, dilate), np.uint8) if dilate > 0 else None

        masks_dir = bundle.stage_dir(self.name) / "masks"
        masks_dir.mkdir(exist_ok=True)
        index: list[dict[str, Any]] = []
        labels_seen: dict[str, int] = {}
        total_frac = 0.0

        for kf in keyframes:
            img = read_image(bundle.artifact_path(kf["path"]))
            h, w = img.shape[:2]
            dets = detector.detect(img, threshold=cfg.masking.detect_threshold)
            mask = np.zeros((h, w), np.uint8)
            n_dyn = 0
            for d in dets:
                if d["label"] not in want:
                    continue
                n_dyn += 1
                labels_seen[d["label"]] = labels_seen.get(d["label"], 0) + 1
                x1, y1, x2, y2 = d["box"]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                mask[y1:y2, x1:x2] = 255
            if kernel is not None and n_dyn:
                mask = cv2.dilate(mask, kernel)

            mpath = masks_dir / (f"{kf['index']:06d}.png")
            if not cv2.imwrite(str(mpath), mask):
                raise RuntimeError(f"s3_masking: failed to write mask {mpath}")
            frac = float((mask > 0).sum()) / float(mask.size)
            total_frac += frac
            index.append({
                "index": kf["index"], "frame": kf["path"], "mask": bundle.relpath(mpath),
                "n_dynamic": n_dyn, "dynamic_frac": round(frac, 4),
            })

        mean_frac = total_frac / len(keyframes)
        masks_ref = bundle.write_json(f"{self.name}/masks.json", {
            "contract_version": 3,
            "convention": "255=dynamic/exclude, 0=static/keep",
            "detector": cfg.masking.detector, "threshold": cfg.masking.detect_threshold,
            "segmentation": None,
            "tracking": None,
            "motion_residual": None,
            "capabilities": _capability_metadata(cfg, decision),
            "n_frames": len(index), "masks": index,
        })

        # measured: less dynamic area => more of the scene is usable static geometry. Capped by the
        # chosen tier's ceiling so a box-mask result never claims pixel-refined confidence.
        measured = round(1.0 - min(1.0, mean_frac), 3)
        return StageResult(
            outputs={"masks": masks_ref, "masks_dir": bundle.relpath(masks_dir)},
            metrics={
                "n_frames": len(index),
                "mean_dynamic_frac": round(mean_frac, 4),
                "labels_seen": labels_seen,
                "masking_tier": decision.tier.name,
                "masking_tier_level": decision.tier.level,
                "masking_not_executed": _capability_metadata(cfg, decision)["not_executed"],
                "semantic_classes_requested": list(cfg.masking.semantic_classes),
            },
            confidence_summary=decision.blend(measured),
            degraded_reason=decision.degraded_reason,
        )
