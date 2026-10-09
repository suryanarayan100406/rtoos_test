"""Rectify supported COLMAP lenses into the pinhole pixel contract consumed by depth/fusion."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..runtime.deps import require
from .depthfit import intrinsics_matrix


def recenter_extrinsic(extrinsic: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Keep world-to-camera rays identical after replacing world coordinates by world - origin.

    Translate in float64 before callers cast to neural float32; doing the subtraction after the cast
    loses sub-metre detail at UTM northings. Returns a new matrix, leaving the CRS input untouched.
    """
    ext = np.asarray(extrinsic, dtype=np.float64).copy()
    ext[:3, 3] += ext[:3, :3] @ np.asarray(origin, dtype=np.float64)
    return ext


def rectification_maps(intrinsics: dict):
    cv2 = require("cv2", purpose="camera distortion rectification")
    model = intrinsics["model"].upper()
    params = intrinsics["params"]
    K = intrinsics_matrix(model, params)
    width, height = intrinsics["width"], intrinsics["height"]
    output: dict[str, Any] = {"model": "PINHOLE", "width": width, "height": height,
              "params": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]}
    if model in {"PINHOLE", "SIMPLE_PINHOLE"}:
        return None, output
    if model == "SIMPLE_RADIAL":
        distortion = np.asarray([params[3], 0, 0, 0, 0], dtype=float)
    elif model == "RADIAL":
        distortion = np.asarray([params[3], params[4], 0, 0, 0], dtype=float)
    elif model in {"OPENCV", "FULL_OPENCV"}:
        distortion = np.asarray(params[4:], dtype=float)
    elif model in {"OPENCV_FISHEYE", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        coeff = params[4:] if model == "OPENCV_FISHEYE" else params[3:]
        distortion = np.zeros(4, dtype=float)
        distortion[:len(coeff)] = coeff
        maps = cv2.fisheye.initUndistortRectifyMap(K, distortion, np.eye(3), K, (width, height), cv2.CV_32FC1)
        return maps, output
    else:
        raise ValueError(f"unsupported lens '{model}'; rectification is required for pinhole fusion")
    maps = cv2.initUndistortRectifyMap(K, np.asarray(distortion), np.eye(3), K, (width, height), cv2.CV_32FC1)
    return maps, output


def rectify_views(bundle, keyframes: list[dict], intrinsics: dict) -> tuple[list[dict], dict]:
    from pathlib import Path

    from ..io.video import read_image, save_frame

    maps, pinhole = rectification_maps(intrinsics)
    if maps is None:
        return keyframes, pinhole
    cv2 = require("cv2", purpose="rectified image generation")
    out = bundle.stage_dir("s6_global") / "rectified"
    out.mkdir(exist_ok=True)
    width, height = intrinsics["width"], intrinsics["height"]
    # Do not allow a neural model to reconstruct the black padding created by lens rectification.
    map_x, map_y = maps
    valid = (np.isfinite(map_x) & np.isfinite(map_y)
             & (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1))
    valid_path = bundle.stage_dir("s6_global") / "valid_pixels.png"
    save_frame(valid.astype(np.uint8) * 255, valid_path)
    views = []
    for frame in keyframes:
        image = read_image(bundle.artifact_path(frame["path"]))
        if image.shape[:2] != (intrinsics["height"], intrinsics["width"]):
            raise ValueError("keyframe image size differs from camera calibration")
        path = out / Path(frame["path"]).name
        save_frame(cv2.remap(image, *maps, cv2.INTER_LINEAR), path)
        views.append({**frame, "path": bundle.relpath(path), "source_path": frame["path"], "rectified": True,
                      "valid_pixels": bundle.relpath(valid_path)})
    return views, pinhole
