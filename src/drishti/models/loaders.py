"""Model loading helpers: license gate, device selection, and registry-name -> weights mapping.

Every model load goes through the registry's license gate ([registry.py]) before any weights are
touched, so a non-permissive model fails *before* download. Loads are guarded (torch/transformers are
optional extras) and honor the offline model cache.
"""
from __future__ import annotations

from pathlib import Path

from ..bundle.hashing import sha256_file
from ..runtime.context import Runtime
from ..runtime.deps import require
from .registry import ModelSpec

# Registry name -> HuggingFace repo id for the shipped, permissive checkpoints.
HF_REPO_IDS = {
    "depth_anything_v2_small": "depth-anything/Depth-Anything-V2-Small-hf",
    "depth_anything_v2_base": "depth-anything/Depth-Anything-V2-Base-hf",
    "rt_detr": "PekingU/rtdetr_r50vd",
    "da3_small": "depth-anything/DA3-SMALL",
    "da3_base": "depth-anything/DA3-BASE",
    "da3_metric_large": "depth-anything/DA3METRIC-LARGE",
    "mapanything_apache": "facebook/map-anything-apache",
}


def gate(rt: Runtime, name: str) -> ModelSpec:
    """Resolve a model spec through the license gate. Raises LicenseError if not permissive."""
    return rt.registry.resolve(name)


def hf_repo_id(name: str) -> str:
    if name not in HF_REPO_IDS:
        raise KeyError(
            f"no HuggingFace repo mapped for model '{name}'. Add it to HF_REPO_IDS "
            f"(and confirm the checkpoint license in the registry)."
        )
    return HF_REPO_IDS[name]


def torch_device(rt: Runtime) -> str:
    """'cuda' when this process has a CUDA GPU (the cloud T4 session), else 'cpu'."""
    return "cuda" if rt.detected.cuda else "cpu"


def cache_dir_for(rt: Runtime) -> Path:
    d = rt.registry.cache_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_path(rt: Runtime, name: str) -> Path:
    """Resolve a pinned, checksum-verified local checkpoint; downloads require explicit configuration."""
    spec = gate(rt, name)
    if not spec.checkpoint or not spec.commit_or_tag or not spec.sha256:
        raise ValueError(f"{name}: checkpoint revision/checksum is not pinned; acquire a verified registry entry")
    hub = require("huggingface_hub", purpose="pinned model cache resolution")
    path = Path(hub.snapshot_download(
        spec.checkpoint, revision=spec.commit_or_tag, cache_dir=str(cache_dir_for(rt)),
        local_files_only=not rt.cfg.models.allow_download,
        allow_patterns=["*.json", "model.safetensors"],
    ))
    weights = path / "model.safetensors"
    actual = sha256_file(weights)
    if actual != spec.sha256:
        raise ValueError(f"{name}: model checksum mismatch: expected {spec.sha256}, got {actual}")
    return path
