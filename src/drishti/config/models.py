"""Typed configuration models (Pydantic v2).

Every tunable in the pipeline is declared here and sourced from configs/*.yaml + overrides — there
are no magic numbers in stage code (AGENTS.md §5, no-hardcode policy). Unknown keys and type errors
fail at load, so a typo in a YAML file is an error, not a silent default.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Base(BaseModel):
    # Reject unknown keys so config typos fail loudly instead of being silently ignored.
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- run / compute
class RunCfg(_Base):
    output_root: str = "runs"
    seed: int = 1234
    profile: str = "balanced"


class ComputeCfg(_Base):
    placement: Literal["auto", "force_local", "force_cloud"] = "auto"
    local_vram_gb: float = 0.0
    cloud_vram_gb: float = 16.0
    prefer_openvino: bool = True
    max_workers: int = 0  # 0 => auto


# --------------------------------------------------------------------------- S0 / S1
class IngestCfg(_Base):
    target_fps: float = 4.0
    max_frames: int = 0
    time_offset_s: float = 0.0


class KeyframeCfg(_Base):
    method: Literal["parallax", "fixed_stride", "feature_disp"] = "parallax"
    min_parallax_px: float = 8.0
    stride: int = 5
    max_keyframes: int = 0


class FrameQACfg(_Base):
    blur_min_varlap: float = 100.0
    exposure_low: float = 0.02
    exposure_high: float = 0.02
    keyframe: KeyframeCfg = Field(default_factory=KeyframeCfg)


# --------------------------------------------------------------------------- S2 poses + spine
class PosesCfg(_Base):
    sfm_backend: Literal["colmap"] = "colmap"
    use_glomap: bool = True
    matcher: Literal["sequential", "exhaustive", "vocab_tree"] = "sequential"
    self_calibrate_intrinsics: bool = True
    min_track_len: int = 3


class SpineCfg(_Base):
    gnss_sigma_m: float = 2.5
    gnss_sigma_v_m: float = 5.0
    use_imu: Literal["auto", "on", "off"] = "auto"
    use_baro: Literal["auto", "on", "off"] = "auto"
    align: Literal["umeyama"] = "umeyama"


class CrsCfg(_Base):
    mode: Literal["derive_from_gps", "epsg"] = "derive_from_gps"
    epsg: int | None = None
    vertical: Literal["ellipsoidal", "orthometric"] = "ellipsoidal"

    @field_validator("epsg")
    @classmethod
    def _epsg_range(cls, v: int | None) -> int | None:
        if v is not None and not (1024 <= v <= 32767):
            raise ValueError(f"epsg {v} out of the EPSG code range")
        return v


# --------------------------------------------------------------------------- S3 / S4
class MaskingCfg(_Base):
    detector: str = "rt_detr"
    segmenter: str = "sam2"
    tracker: str = "bytetrack"
    motion: str = "raft"
    detect_threshold: float = 0.4
    dynamic_dilate_px: int = 8
    dynamic_classes: list[str] = Field(default_factory=list)
    semantic_classes: list[str] = Field(default_factory=list)


class FeedforwardCfg(_Base):
    enabled: bool = False
    model: str = "mapanything_apache"
    window: int = 8


class DepthCfg(_Base):
    model: str = "depth_anything_v2_small"
    metric_refiner: str = "metric3d_v2"
    input_long_side: int = 1024
    min_metric_anchors: int = 8
    geometry_window: int = 8
    scale_align: Literal["sparse", "none"] = "sparse"
    feedforward_geometry: FeedforwardCfg = Field(default_factory=FeedforwardCfg)


# --------------------------------------------------------------------------- S7 / S8
class GaussianCfg(_Base):
    iters: int = 7000
    sh_degree: int = 2
    regularize: list[str] = Field(default_factory=lambda: ["depth", "normal", "confidence"])
    max_gaussians: int = 2_000_000
    view_batch: int = 1
    checkpoint_every: int = 500


class TileCfg(_Base):
    enabled: bool = True
    tile_m: float = 40.0
    overlap_m: float = 8.0


class DenseCfg(_Base):
    method: Literal["tsdf", "gaussian"] = "tsdf"
    tsdf_voxel_m: float = 0.05
    depth_trunc_m: float = 100.0  # ignore depth beyond this (unreliable far field)
    # Vertical margin added to the sparse-point surface envelope when clipping depth in S7 tiled fusion.
    # Big enough to keep real structure + depth noise at the true surface; small enough to drop far-field
    # monocular flyers that otherwise inflate per-tile RAM. Set huge to effectively disable the Z clip.
    ground_band_margin_m: float = 12.0
    min_local_surface_points: int = 3
    max_vertical_slab_m: float = Field(default=10.0, gt=0)
    gaussian: GaussianCfg = Field(default_factory=GaussianCfg)
    tile: TileCfg = Field(default_factory=TileCfg)


class MeshCfg(_Base):
    method: Literal["poisson", "2dgs", "sugar"] = "poisson"
    poisson_depth: int = 11
    density_quantile: float = 0.03   # crop this lowest fraction of Poisson vertex densities
    normal_max_nn: int = 30          # kNN for normal estimation
    # Downsample voxel (m) for the meshing INPUT only (the 5 cm dense.ply is untouched). Feeding Poisson
    # points finer than its octree leaf (scene_extent / 2**poisson_depth) wastes RAM without adding mesh
    # detail — a 300 m scene at depth 11 resolves ~16 cm, so a 100 M-point 5 cm aerial cloud OOMs the mesh
    # for nothing. 0.0 = auto (match the octree leaf, clamped >= dense voxel so we never invent detail);
    # >0 forces a voxel; <=dense-voxel effectively disables the reduction.
    mesh_voxel_m: float = 0.0
    # Above this many points, orient normals with the O(N) aerial up-prior instead of the O(N*kNN)
    # tangent-plane MST (orient_normals_consistent_tangent_plane), which builds a graph over every point
    # and exhausts host RAM on 10 M+ -point clouds. Up-orientation is correct for nadir 2.5D terrain.
    consistent_normals_max_points: int = 5_000_000
    texture: Literal["mvs_texturing", "best_view"] = "mvs_texturing"
    texture_atlas_size: int = Field(default=2048, ge=256, le=16384)
    texture_visibility_tolerance_m: float = Field(default=1.0, gt=0)
    emit_confidence: bool = True
    flag_inferred: bool = True


# --------------------------------------------------------------------------- S9 / S10
class GeoCfg(_Base):
    dsm_res_m: float = 0.10
    dtm_res_m: float = 0.25
    ortho_res_m: float = 0.05
    dtm_ground_window_m: float = 3.0   # structuring-element size for the morphological ground filter
    ground_classifier: Literal["pdal_smrf", "morphological_opening"] = "pdal_smrf"
    pdal_smrf_slope: float = 0.15
    pdal_smrf_threshold: float = 0.5
    pdal_smrf_scalar: float = 1.25


class ExportCfg(_Base):
    formats: list[str] = Field(default_factory=lambda: ["obj", "ply", "las", "geotiff", "gltf", "fbx"])
    also: list[str] = Field(default_factory=lambda: ["laz", "3dtiles", "potree"])
    fbx_via: Literal["blender_headless"] = "blender_headless"
    point_format: str = "las_1_4"


class ReportCfg(_Base):
    check_points: str | None = None
    format: list[str] = Field(default_factory=lambda: ["html", "json"])


class ModelsCfg(_Base):
    cache_dir: str = "models_cache"
    allow_reference_only: bool = False
    allow_download: bool = False  # acquisition must be explicitly enabled; missions default offline


# --------------------------------------------------------------------------- root
class DrishtiConfig(_Base):
    run: RunCfg = Field(default_factory=RunCfg)
    compute: ComputeCfg = Field(default_factory=ComputeCfg)
    ingest: IngestCfg = Field(default_factory=IngestCfg)
    frameqa: FrameQACfg = Field(default_factory=FrameQACfg)
    poses: PosesCfg = Field(default_factory=PosesCfg)
    spine: SpineCfg = Field(default_factory=SpineCfg)
    crs: CrsCfg = Field(default_factory=CrsCfg)
    masking: MaskingCfg = Field(default_factory=MaskingCfg)
    depth: DepthCfg = Field(default_factory=DepthCfg)
    dense: DenseCfg = Field(default_factory=DenseCfg)
    mesh: MeshCfg = Field(default_factory=MeshCfg)
    geo: GeoCfg = Field(default_factory=GeoCfg)
    export: ExportCfg = Field(default_factory=ExportCfg)
    report: ReportCfg = Field(default_factory=ReportCfg)
    models: ModelsCfg = Field(default_factory=ModelsCfg)


# --------------------------------------------------------------------------- dataset descriptor
class CSVMapping(_Base):
    time: str = "timestamp"
    lat: str = "latitude"
    lon: str = "longitude"
    alt: str = "abs_alt"
    yaw: str | None = None
    pitch: str | None = None
    roll: str | None = None


class TelemetrySpec(_Base):
    format: Literal["dji_srt", "csv", "mavlink", "exif"]
    path: str | None = None
    csv: CSVMapping | None = None


class OptionalInputs(_Base):
    imu: str | None = None
    baro: str | None = None
    intrinsics: str | None = None
    rtk: str | None = None


class DatasetDescriptor(_Base):
    """The input contract: how a user points DRISHTI at a real recording."""
    name: str
    video: str
    telemetry: TelemetrySpec
    optional: OptionalInputs = Field(default_factory=OptionalInputs)
    # crs/report here override the same sections in the main config (merged during load).
    crs: CrsCfg | None = None
    report: ReportCfg | None = None
