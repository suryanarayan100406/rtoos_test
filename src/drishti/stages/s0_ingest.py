"""S0 — Ingest & time-align telemetry.

Offline realization of capture+sync: decode the recorded clip, parse and time-align the GPS/flight
telemetry, and derive the working CRS from the GPS track. Mandatory inputs (video, telemetry) missing
=> loud failure, never a placeholder (AGENTS.md §5).
"""
from __future__ import annotations

from typing import Any

from ..bundle import Bundle, CrsInfo
from ..config import DrishtiConfig, TelemetrySpec
from ..geo.crs import derive_crs_from_track
from ..io.telemetry import parse_telemetry
from ..runtime import ComputeNeed, Runtime
from .base import Stage, StageResult, register_stage


@register_stage
class IngestStage(Stage):
    name = "s0_ingest"
    requires = ()
    compute = ComputeNeed(cpu_only=True)  # decode + parse; OpenVINO not needed

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"ingest": cfg.ingest.model_dump(), "crs": cfg.crs.model_dump()}

    def extra_input_hashes(self, bundle: Bundle, cfg: DrishtiConfig) -> list[str]:
        from ..bundle.hashing import sha256_file

        desc = bundle.manifest.dataset_snapshot or {}
        hashes: list[str] = []
        for p in [desc.get("video"), (desc.get("telemetry") or {}).get("path")]:
            try:
                if p:
                    hashes.append(sha256_file(p))
            except OSError:
                hashes.append(f"missing:{p}")
        return hashes

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        desc = bundle.manifest.dataset_snapshot
        if not desc:
            raise ValueError(
                "s0_ingest: no dataset descriptor in the bundle. Run via "
                "`drishti run --dataset <descriptor.yaml>`."
            )

        # --- register inputs (loud fail on missing mandatory files) ---
        bundle.register_input(desc.get("video"), role="video", required=True)
        tspec = TelemetrySpec.model_validate(desc["telemetry"])
        bundle.register_input(tspec.path, role="gps", required=True)
        opt = desc.get("optional") or {}
        for role in ("imu", "baro", "intrinsics", "rtk"):
            bundle.register_input(opt.get(role), role=role, required=False)
        bundle.save_manifest()

        # --- telemetry ---
        tel = parse_telemetry(tspec)
        tel_ref = bundle.write_json(f"{self.name}/telemetry.json", tel.to_jsonable())

        # --- CRS derived from the GPS track (never hardcoded) ---
        override = cfg.crs.epsg if cfg.crs.mode == "epsg" else None
        crs = derive_crs_from_track(tel.lons(), tel.lats(), override_epsg=override)
        bundle.manifest.crs = CrsInfo(
            epsg=crs.epsg, derived_from=crs.derived_from, vertical=cfg.crs.vertical
        )

        # --- frames: decode + sample + attach interpolated GPS ---
        # imported lazily so telemetry/CRS work even before video deps are installed
        from ..io.video import iter_frames, probe_video, save_frame

        info = probe_video(desc["video"])
        frames_dir = bundle.stage_dir(self.name) / "frames"
        frames_dir.mkdir(exist_ok=True)
        frames: list[dict[str, Any]] = []
        offset = cfg.ingest.time_offset_s
        for idx, (src_index, t, img) in enumerate(
            iter_frames(desc["video"], cfg.ingest.target_fps, cfg.ingest.max_frames)
        ):
            fpath = frames_dir / f"{idx:06d}.jpg"
            save_frame(img, fpath)
            s = tel.sample_at(t + offset)
            frames.append({
                "index": idx, "src_index": src_index, "t": round(t, 4),
                "path": bundle.relpath(fpath),
                "lat": s.lat, "lon": s.lon, "alt": s.alt,
            })

        if not frames:
            raise ValueError("s0_ingest: no frames decoded from the video (empty or unreadable).")

        frames_ref = bundle.write_json(f"{self.name}/frames.json", {
            "video": {"width": info.width, "height": info.height, "fps": info.fps,
                      "duration_s": info.duration_s, "n_frames": info.n_frames},
            "sampled_fps": cfg.ingest.target_fps, "n_sampled": len(frames), "frames": frames,
        })

        # coverage confidence: fraction of the video timespan covered by telemetry
        coverage = min(1.0, tel.duration / info.duration_s) if info.duration_s else 0.0
        return StageResult(
            outputs={"telemetry": tel_ref, "frames": frames_ref, "frames_dir": bundle.relpath(frames_dir)},
            metrics={
                "n_telemetry_samples": len(tel),
                "telemetry_fields": sorted(tel.fields_present),
                "n_frames_sampled": len(frames),
                "video_duration_s": round(info.duration_s, 2),
                "crs_epsg": crs.epsg,
                "crs_derived_from": crs.derived_from,
            },
            confidence_summary=round(coverage, 3),
        )
