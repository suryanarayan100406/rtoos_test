"""GLOMAP global-refinement adapter.

GLOMAP is an optional native executable.  This adapter only reports success after the executable returns
zero *and* a readable COLMAP model is found and parsed with pycolmap.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GlomapPose:
    name: str
    rotation: np.ndarray
    center: np.ndarray


@dataclass
class GlomapResult:
    poses: list[GlomapPose]
    sparse_points: np.ndarray
    intrinsics: dict
    model_path: Path
    stdout: str
    stderr: str


def _find_model(output_dir: Path) -> Path | None:
    candidates = [output_dir, *sorted(output_dir.rglob("*"))]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "cameras.bin").is_file() and (candidate / "images.bin").is_file():
            return candidate
    return None


def run_glomap(database_path: Path, image_dir: Path, output_dir: Path) -> GlomapResult:
    """Run GLOMAP's mapper and parse its resulting registered camera poses."""
    binary = shutil.which("glomap")
    if binary is None:
        raise FileNotFoundError("glomap executable not found on PATH")
    output_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [binary, "mapper", "--database_path", str(database_path), "--image_path", str(image_dir),
         "--output_path", str(output_dir)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"GLOMAP mapper failed with exit code {proc.returncode}: "
            f"{proc.stderr[-1000:] or proc.stdout[-1000:]}"
        )
    model_path = _find_model(output_dir)
    if model_path is None:
        raise RuntimeError(f"GLOMAP exited successfully but wrote no COLMAP model under {output_dir}")

    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model_path))
    poses: list[GlomapPose] = []
    for image in reconstruction.images.values():
        cfw = image.cam_from_world
        if callable(cfw):
            cfw = cfw()
        rotation = np.asarray(cfw.rotation.matrix(), dtype=float)
        translation = np.asarray(cfw.translation, dtype=float).reshape(3)
        poses.append(GlomapPose(
            name=image.name,
            rotation=rotation,
            center=-rotation.T @ translation,
        ))
    if len(poses) < 3:
        raise RuntimeError(f"GLOMAP produced only {len(poses)} readable poses; need at least 3")
    cameras = list(reconstruction.cameras.values())
    if len(cameras) != 1:
        raise ValueError("GLOMAP returned multiple calibration models; candidate needs per-view calibration")
    cam = cameras[0]
    intr = {"model": cam.model.name, "width": int(cam.width), "height": int(cam.height),
            "params": list(map(float, cam.params))}
    points = np.asarray([p.xyz for p in reconstruction.points3D.values()], dtype=float).reshape(-1, 3)
    return GlomapResult(poses=poses, sparse_points=points, intrinsics=intr, model_path=model_path,
                        stdout=proc.stdout[-4000:], stderr=proc.stderr[-4000:])
