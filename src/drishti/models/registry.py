"""Model registry + license gate.

Every model DRISHTI can use is pinned here with its license. The shipped pipeline refuses to load a
non-permissive or reference-only model unless explicitly overridden (models.allow_reference_only).
This encodes the binding licensing posture from docs/03-TECHNOLOGY-STACK.md and AGENTS.md §5.

Licenses below reflect the shipped checkpoints/components; where a family mixes licenses (e.g. Depth
Anything V2 small/base are Apache-2.0 but large is CC-BY-NC), each variant is a separate entry.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

# Permissive licenses allowed in the shipped path.
LICENSE_ALLOWLIST = {
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC-BY-4.0",
}


class LicenseError(RuntimeError):
    """Raised when the pipeline tries to load a non-permissive / reference-only model."""


class ModelNotRegistered(KeyError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: str  # depth | detect | segment | track | flow | geometry | recon | pose
    license: str
    runtimes: tuple[str, ...] = ("torch",)  # torch | openvino | native
    version: str = "unpinned"
    sha256: str | None = None
    source: str = ""
    vram_gb: float = 0.0
    reference_only: bool = False  # true => benchmark/reference only, never shipped
    notes: str = ""
    checkpoint: str | None = None
    checkpoint_license: str | None = None
    commercial_status: str = "unverified"
    commit_or_tag: str | None = None
    ram_gb: float = 0.0
    precision: str = "fp32"
    devices: tuple[str, ...] = ("cpu",)
    download_source: str | None = None
    offline_compatible: bool = True

    @property
    def permissive(self) -> bool:
        return (
            self.license in LICENSE_ALLOWLIST and not self.reference_only
            and (self.checkpoint_license is None or self.checkpoint_license in LICENSE_ALLOWLIST)
        )


# --------------------------------------------------------------------------- the registry
_SPECS: dict[str, ModelSpec] = {
    # --- poses -------------------------------------------------------------
    "colmap": ModelSpec("colmap", "pose", "BSD-3-Clause", ("native",), source="colmap.github.io"),
    "glomap": ModelSpec("glomap", "pose", "BSD-3-Clause", ("native",), source="github.com/colmap/glomap"),
    # --- metric depth (shipped: permissive small; base is reference-only) ---
    "depth_anything_v2_small": ModelSpec(
        "depth_anything_v2_small", "depth", "Apache-2.0", ("openvino", "torch"), vram_gb=2.0,
        source="github.com/DepthAnything/Depth-Anything-V2", version="hf-2024-07-05",
        checkpoint="depth-anything/Depth-Anything-V2-Small-hf", checkpoint_license="Apache-2.0",
        commercial_status="commercial-friendly",
        sha256="3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70",
        ram_gb=4.0, devices=("cpu", "cuda"),
        download_source="huggingface.co/depth-anything/Depth-Anything-V2-Small-hf"),
    "depth_anything_v2_base": ModelSpec(
        "depth_anything_v2_base", "depth", "CC-BY-NC-4.0", ("openvino", "torch"), vram_gb=4.0,
        source="github.com/DepthAnything/Depth-Anything-V2", version="hf-2024-07-05",
        checkpoint="depth-anything/Depth-Anything-V2-Base-hf", checkpoint_license="CC-BY-NC-4.0",
        commercial_status="non-commercial", reference_only=True,
        notes="Upstream HF base checkpoint is CC-BY-NC; reference only."),
    "depth_anything_v2_large": ModelSpec(
        "depth_anything_v2_large", "depth", "CC-BY-NC-4.0", ("torch",), vram_gb=8.0,
        reference_only=True, notes="large ckpt is non-commercial; not shipped"),
    "metric3d_v2": ModelSpec(
        "metric3d_v2", "depth", "BSD-2-Clause", ("torch",), vram_gb=6.0,
        source="github.com/YvanYin/Metric3D"),
    "unidepth_v2": ModelSpec(
        "unidepth_v2", "depth", "CC-BY-NC-4.0", ("torch",), vram_gb=6.0, reference_only=True,
        notes="non-commercial; reference only"),
    # --- masking / perception ---------------------------------------------
    "rt_detr": ModelSpec("rt_detr", "detect", "Apache-2.0", ("openvino", "torch"), vram_gb=3.0,
                         source="github.com/lyuwenyu/RT-DETR", notes="permissive alternative to AGPL YOLO"),
    "sam2": ModelSpec("sam2", "segment", "Apache-2.0", ("torch",), vram_gb=5.0,
                      source="github.com/facebookresearch/sam2"),
    "bytetrack": ModelSpec("bytetrack", "track", "MIT", ("native",), source="github.com/ifzhang/ByteTrack"),
    "raft": ModelSpec("raft", "flow", "BSD-3-Clause", ("torch",), vram_gb=3.0,
                      source="github.com/princeton-vl/RAFT"),
    # --- feed-forward geometry (OPTIONAL; blocked until license verified) --
    "mapanything": ModelSpec(
        "mapanything", "geometry", "verify-before-ship", ("torch",), vram_gb=12.0,
        reference_only=True, notes="OPTIONAL enhancement; confirm license before shipping"),
    "mapanything_apache": ModelSpec(
        "mapanything_apache", "geometry", "Apache-2.0", ("torch",), vram_gb=12.0, ram_gb=16.0,
        source="github.com/facebookresearch/map-anything",
        checkpoint="facebook/map-anything-apache", checkpoint_license="Apache-2.0",
        commercial_status="commercial-friendly", precision="bf16/fp16",
        devices=("cuda",), download_source="huggingface.co/facebook/map-anything-apache",
        version="hf-2026-02-04",
        sha256="fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201",
        notes="Current upstream Apache checkpoint; optional geometry backend, not default until aerial validation."),
    "da3_small": ModelSpec(
        "da3_small", "geometry", "Apache-2.0", ("torch",), vram_gb=4.0, ram_gb=8.0,
        source="github.com/ByteDance-Seed/Depth-Anything-3",
        checkpoint="depth-anything/DA3-SMALL", checkpoint_license="Apache-2.0",
        commercial_status="commercial-friendly", precision="fp16", devices=("cuda", "cpu"),
        download_source="huggingface.co/depth-anything/DA3-SMALL", version="hf-2025-11-13",
        sha256="364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf",
        notes="Multi-view/pose/depth family; validate aerial behavior before production default."),
    "da3_base": ModelSpec(
        "da3_base", "geometry", "Apache-2.0", ("torch",), vram_gb=6.0, ram_gb=12.0,
        source="github.com/ByteDance-Seed/Depth-Anything-3",
        checkpoint="depth-anything/DA3-BASE", checkpoint_license="Apache-2.0",
        commercial_status="commercial-friendly", precision="fp16", devices=("cuda", "cpu"),
        download_source="huggingface.co/depth-anything/DA3-BASE", version="hf-2025-11-15",
        sha256="e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5",
        notes="Multi-view/pose/depth family; validate aerial behavior before production default."),
    "da3_metric_large": ModelSpec(
        "da3_metric_large", "depth", "Apache-2.0", ("torch",), vram_gb=8.0, ram_gb=12.0,
        source="github.com/ByteDance-Seed/Depth-Anything-3",
        checkpoint="depth-anything/DA3METRIC-LARGE", checkpoint_license="Apache-2.0",
        commercial_status="commercial-friendly", precision="fp16", devices=("cuda", "cpu"),
        download_source="huggingface.co/depth-anything/DA3METRIC-LARGE", version="hf-2025-11-13",
        sha256="bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776",
        notes="Metric monocular depth candidate; its focal-length scaling contract must be honored."),
    "pi3": ModelSpec(
        "pi3", "geometry", "verify-before-ship", ("torch",), vram_gb=10.0, reference_only=True,
        notes="OPTIONAL enhancement; confirm license before shipping"),
    "vggt": ModelSpec(
        "vggt", "geometry", "commercial-military-excluded", ("torch",), vram_gb=16.0,
        reference_only=True, notes="license excludes military use; reference only"),
    "dust3r": ModelSpec("dust3r", "geometry", "CC-BY-NC-4.0", ("torch",), reference_only=True),
    "mast3r": ModelSpec("mast3r", "geometry", "CC-BY-NC-4.0", ("torch",), reference_only=True),
    # --- dense / mesh ------------------------------------------------------
    "gsplat": ModelSpec("gsplat", "recon", "Apache-2.0", ("torch",), vram_gb=10.0,
                        source="github.com/nerfstudio-project/gsplat"),
    "open3d": ModelSpec("open3d", "recon", "MIT", ("native",), source="open3d.org"),
    "mvs_texturing": ModelSpec("mvs_texturing", "recon", "BSD-3-Clause", ("native",),
                              source="github.com/nmoehrle/mvs-texturing"),
}

# Immutable checkpoint revisions and SHA-256 values retrieved from upstream HF metadata, not guessed.
for _name, _revision in {
    "depth_anything_v2_small": "5426e4f0f36572d16453bbda7a8389317b1bef99",
    "da3_small": "e08cab65ca0ec38e7826075418411ab90cab4da3",
    "da3_base": "f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
    "da3_metric_large": "4010e39f3634a45bc60553321fb49fb760bd594e",
    "mapanything_apache": "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a",
}.items():
    _SPECS[_name] = replace(_SPECS[_name], commit_or_tag=_revision)
_SPECS["rt_detr"] = replace(
    _SPECS["rt_detr"], checkpoint="PekingU/rtdetr_r50vd",
    commit_or_tag="df939e661d8c52e80608d1ec566561aabd25a4e7", version="hf-2025-02-06",
    sha256="5263d5521eff3e356f6cd8a371fd5dfb891725beda5f713674f79669115cdc64",
    checkpoint_license="Apache-2.0", commercial_status="commercial-friendly", devices=("cpu", "cuda"),
)


def all_specs() -> dict[str, ModelSpec]:
    return dict(_SPECS)


class ModelRegistry:
    """Resolves model specs and enforces the license gate."""

    def __init__(self, cache_dir: str | Path = "models_cache", allow_reference_only: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.allow_reference_only = allow_reference_only

    def spec(self, name: str) -> ModelSpec:
        try:
            return _SPECS[name]
        except KeyError as exc:
            raise ModelNotRegistered(
                f"model '{name}' is not in the registry; register it (with a verified license) first"
            ) from exc

    def check_license(self, spec: ModelSpec) -> None:
        if spec.permissive:
            return
        if self.allow_reference_only:
            return  # explicit opt-in (e.g. offline benchmarking), never the default
        raise LicenseError(
            f"refusing to load '{spec.name}': license '{spec.license}'"
            f"{' (reference_only)' if spec.reference_only else ''} is not permissive. "
            f"It may be used for reference/benchmarking only. Set models.allow_reference_only=true "
            f"to override for non-shipped use."
        )

    def resolve(self, name: str) -> ModelSpec:
        """Return the spec after passing the license gate (raises otherwise)."""
        spec = self.spec(name)
        self.check_license(spec)
        return spec

    def cache_path(self, name: str) -> Path:
        return self.cache_dir / name
