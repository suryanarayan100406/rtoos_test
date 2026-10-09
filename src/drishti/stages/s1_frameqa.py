"""S1 — Frame QA & keyframe selection.

Score every sampled frame for sharpness (variance of Laplacian) and exposure, drop unusable frames,
and select keyframes by parallax so downstream SfM sees well-separated, well-exposed views. All real
measurements on decoded pixels — no synthetic scores (AGENTS.md §5).
"""
from __future__ import annotations

from typing import Any

from ..bundle import Bundle
from ..config import DrishtiConfig
from ..frameqa import (
    exposure_clip_fractions,
    is_accepted,
    select_by_displacement,
    select_fixed_stride,
    to_gray,
    variance_of_laplacian,
)
from ..runtime import ComputeNeed, Runtime
from .base import Stage, StageResult, register_stage


def _median_orb_displacement(img_a, img_b, cv2, np) -> float:
    """Median pixel displacement of ORB feature matches between two frames (0 if too few)."""
    orb = cv2.ORB_create(nfeatures=1500)
    ka, da = orb.detectAndCompute(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < 8:
        return 0.0
    disp = [
        float(np.hypot(ka[m.queryIdx].pt[0] - kb[m.trainIdx].pt[0],
                       ka[m.queryIdx].pt[1] - kb[m.trainIdx].pt[1]))
        for m in matches
    ]
    return float(np.median(disp))


@register_stage
class FrameQAStage(Stage):
    name = "s1_frameqa"
    requires = ("s0_ingest",)
    compute = ComputeNeed(cpu_only=True)

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"frameqa": cfg.frameqa.model_dump()}

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        import numpy as np

        from ..geo.crs import haversine_m
        from ..io.video import read_image

        frames_doc = bundle.read_json("s0_ingest/frames.json")
        frames = frames_doc["frames"]
        if not frames:
            raise ValueError("s1_frameqa: no frames from s0_ingest.")

        qa = cfg.frameqa
        scored: list[dict[str, Any]] = []
        for fr in frames:
            img = read_image(bundle.artifact_path(fr["path"]))
            gray = to_gray(img)
            varlap = variance_of_laplacian(gray)
            clip_lo, clip_hi = exposure_clip_fractions(gray)
            ok = is_accepted(varlap, clip_lo, clip_hi,
                             qa.blur_min_varlap, qa.exposure_low, qa.exposure_high)
            sharpness_score = min(1.0, varlap / max(qa.blur_min_varlap, 1e-6))
            exposure_score = max(
                0.0,
                1.0 - max(
                    clip_lo / max(qa.exposure_low, 1e-6),
                    clip_hi / max(qa.exposure_high, 1e-6),
                ),
            )
            scored.append({
                "index": fr["index"], "t": fr["t"], "path": fr["path"],
                "lat": fr.get("lat"), "lon": fr.get("lon"), "alt": fr.get("alt"),
                "varlap": round(varlap, 2),
                "clip_low": round(clip_lo, 4), "clip_high": round(clip_hi, 4),
                "quality_score": round(sharpness_score * exposure_score, 4),
                "geometry_score": 0.0,
                "coverage_score": 0.0,
                "selection_reason": "qa_rejected" if not ok else "candidate",
                "accepted": bool(ok),
            })

        accepted = [s for s in scored if s["accepted"]]
        if not accepted:
            raise ValueError(
                f"s1_frameqa: every frame failed QA (blur_min_varlap={qa.blur_min_varlap}). "
                "Footage may be too blurry/dark, or thresholds too strict."
            )

        acc_idx = [s["index"] for s in accepted]
        method = qa.keyframe.method
        if method == "fixed_stride":
            key_idx = select_fixed_stride(acc_idx, qa.keyframe.stride, qa.keyframe.max_keyframes)
            disps = [0.0] * len(accepted)
        else:
            # parallax / feature_disp: measure real ORB displacement between consecutive accepted frames
            from ..runtime.deps import require

            cv2 = require("cv2", purpose="parallax keyframe selection")
            disps = [0.0]
            prev = read_image(bundle.artifact_path(accepted[0]["path"]))
            for s in accepted[1:]:
                cur = read_image(bundle.artifact_path(s["path"]))
                disps.append(_median_orb_displacement(prev, cur, cv2, np))
                prev = cur
            key_idx = select_by_displacement(
                acc_idx, disps, qa.keyframe.min_parallax_px, qa.keyframe.max_keyframes
            )

        # Persist measured geometry/coverage evidence for every candidate. Geometry is the actual ORB
        # displacement used by the selector; coverage is cumulative GNSS-track progress, not a proxy
        # generated from the frame index. Pair equal-length slices for the n-1 adjacent segments.
        total_track = sum(
            haversine_m(float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"]))
            for a, b in zip(accepted[:-1], accepted[1:], strict=True)
            if None not in (a.get("lat"), a.get("lon"), b.get("lat"), b.get("lon"))
        )
        cumulative = 0.0
        for i, frame in enumerate(accepted):
            if i:
                a, b = accepted[i - 1], frame
                if None not in (a.get("lat"), a.get("lon"), b.get("lat"), b.get("lon")):
                    cumulative += haversine_m(float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"]))
            frame["geometry_score"] = round(
                min(1.0, disps[i] / max(qa.keyframe.min_parallax_px, 1e-6)), 4
            )
            frame["coverage_score"] = round(cumulative / max(total_track, 1e-6), 4)

        key_set = set(key_idx)
        for s in scored:
            s["keyframe"] = s["index"] in key_set
            if s["keyframe"]:
                s["selection_reason"] = (
                    "fixed_stride" if method == "fixed_stride" else "parallax_threshold"
                )
            elif s["accepted"]:
                s["selection_reason"] = "not_selected_by_budget_or_parallax"

        qa_ref = bundle.write_json(f"{self.name}/frameqa.json", {
            "method": method, "n_frames": len(scored), "n_accepted": len(accepted),
            "n_keyframes": len(key_idx), "frames": scored,
        })
        key_ref = bundle.write_json(f"{self.name}/keyframes.json", {
            "keyframes": [s for s in scored if s["keyframe"]],
        })

        accept_rate = len(accepted) / len(scored)
        return StageResult(
            outputs={"frameqa": qa_ref, "keyframes": key_ref},
            metrics={
                "n_frames": len(scored),
                "n_accepted": len(accepted),
                "accept_rate": round(accept_rate, 3),
                "n_keyframes": len(key_idx),
                "median_varlap": round(float(np.median([s["varlap"] for s in scored])), 2),
            },
            confidence_summary=round(accept_rate, 3),
            degraded_reason=None if len(key_idx) >= 2 else "fewer than 2 keyframes selected",
        )
