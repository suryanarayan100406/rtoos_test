"""Swappable learned-geometry backends.

The production pipeline keeps the metric spine and artifact contracts independent of a specific neural
model.  These adapters use the official upstream APIs when the optional packages are installed; they do
not synthesize outputs when a checkpoint or runtime is unavailable.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..runtime.context import Runtime
from ..runtime.deps import require
from .loaders import checkpoint_path, gate, torch_device


def as_numpy(value) -> np.ndarray:
    """Detach tensors and transfer from CUDA before NumPy conversion."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def camera_to_world(extrinsic: np.ndarray) -> np.ndarray:
    extrinsic = as_numpy(extrinsic)
    ext = np.eye(4, dtype=np.float64)
    if extrinsic.shape not in {(3, 4), (4, 4)}:
        raise ValueError(f"expected a 3x4 or 4x4 world-to-camera matrix, got {extrinsic.shape}")
    ext[:3, :] = extrinsic[:3, :]
    return np.linalg.inv(ext)


def remap_prediction(prediction, original_K: np.ndarray, height: int, width: int):
    """Map output rays to original pixels, preserving resize/crop calibration and invalid regions."""
    cv2 = require("cv2", purpose="geometry output ray remapping")
    if prediction.intrinsics is None:
        raise ValueError("geometry backend did not return calibrated output intrinsics")
    y, x = np.indices((height, width), dtype=np.float32)
    mapping = np.asarray(prediction.intrinsics) @ np.linalg.inv(original_K)
    source = mapping @ np.stack([x.ravel(), y.ravel(), np.ones(x.size)], axis=0)
    u, v = (source[:2] / source[2:]).astype(np.float32).reshape(2, height, width)
    depth = cv2.remap(prediction.depth, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    confidence = None if prediction.confidence is None else cv2.remap(
        prediction.confidence, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    return depth, confidence


@dataclass
class GeometryPrediction:
    """Canonical output returned by a learned geometry backend."""

    depth: np.ndarray
    confidence: np.ndarray | None = None
    points_world: np.ndarray | None = None
    pose_world_from_camera: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    metric: bool = False
    depth_convention: str = "camera_z"
    confidence_semantics: str = "uncalibrated_model_score"


class GeometryBackend(Protocol):
    """Minimum backend contract used by benchmark and reconstruction stages."""

    name: str

    def predict(
        self,
        image_paths: Sequence[Path],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
    ) -> list[GeometryPrediction]:
        """Predict geometry for an ordered view window."""


class DepthAnything3Backend:
    """Official Depth Anything 3 adapter for Apache model variants."""

    name = "depth_anything_3"

    def __init__(self, rt: Runtime, model_name: str) -> None:
        self.spec = gate(rt, model_name)
        if self.spec.task not in {"depth", "geometry"}:
            raise ValueError(f"model '{model_name}' is not a depth/geometry backend")
        torch = require("torch", purpose="Depth Anything 3 inference")
        da3 = require("depth_anything_3.api", purpose="Depth Anything 3 inference")
        self.device = torch_device(rt)
        model_cls = getattr(da3, "DepthAnything3", None)
        if model_cls is None:
            raise RuntimeError("installed depth_anything_3 package does not expose DepthAnything3")
        self.model = model_cls.from_pretrained(str(checkpoint_path(rt, model_name)), local_files_only=True)
        self.model = self.model.to(device=self.device).eval()
        self._torch = torch
        self.process_res = rt.cfg.depth.input_long_side

    def predict(
        self,
        image_paths: Sequence[Path],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
    ) -> list[GeometryPrediction]:
        if not image_paths:
            return []
        input_intrinsics = intrinsics
        pose_conditioned = extrinsics is not None and len(image_paths) >= 3 and self.spec.task == "geometry"
        with self._torch.no_grad():
            result = self.model.inference(
                [str(p) for p in image_paths],
                extrinsics=extrinsics if pose_conditioned else None,
                intrinsics=intrinsics,
                process_res=self.process_res,
                process_res_method="upper_bound_resize",
                ref_view_strategy="first",
            )
        depths = as_numpy(result.depth)
        if depths.ndim != 3 or len(depths) != len(image_paths):
            raise RuntimeError("Depth Anything 3 returned an unexpected number or shape of depth maps")
        confidences = getattr(result, "conf", None)
        conf_array = None if confidences is None else as_numpy(confidences)
        extrinsics = getattr(result, "extrinsics", None)
        intrinsics = getattr(result, "intrinsics", None)
        predictions: list[GeometryPrediction] = []
        for i in range(len(image_paths)):
            depth = as_numpy(depths[i]).astype(np.float32)
            # DA3 upper_bound_resize does not crop equal-sized recording frames. Input K is the
            # pixel transform for that resize, even when the camera head predicts a different K.
            output_K = None
            if input_intrinsics is not None:
                from PIL import Image
                with Image.open(image_paths[i]) as image:
                    original_w, original_h = image.size
                output_K = as_numpy(input_intrinsics[i]).astype(np.float64).copy()
                output_K[0] *= depth.shape[1] / original_w
                output_K[1] *= depth.shape[0] / original_h
            elif intrinsics is not None:
                output_K = as_numpy(intrinsics[i]).copy()
            metric = pose_conditioned
            if self.spec.name == "da3_metric_large":
                if output_K is None or input_intrinsics is None:
                    raise ValueError("DA3 Metric requires calibrated focal length for output scaling")
                # FAQ: use focal length at the network processing resolution, not original video pixels.
                focal = (output_K[0, 0] + output_K[1, 1]) / 2
                depth = depth * float(focal) / 300.0
                metric = True
            predictions.append(GeometryPrediction(
                depth=depth,
                confidence=None if conf_array is None else as_numpy(conf_array[i]).astype(np.float32),
                pose_world_from_camera=None if extrinsics is None else camera_to_world(extrinsics[i]),
                intrinsics=output_K, metric=metric,
            ))
        return predictions

    def predict_path(self, image_path: Path) -> GeometryPrediction:
        """Run one-frame inference while keeping the loaded checkpoint resident."""
        predictions = self.predict([image_path])
        if len(predictions) != 1:
            raise RuntimeError("Depth Anything 3 returned an unexpected number of predictions")
        return predictions[0]


class MapAnythingBackend:
    """Official MapAnything Apache checkpoint adapter.

    MapAnything accepts mixed image/calibration/pose/depth views and returns metric pointmaps plus
    per-pixel confidence.  The adapter intentionally exposes the native output rather than reducing it
    to a scalar depth, so a future fusion stage can use pointmaps and confidence without another API
    migration.
    """

    name = "mapanything"

    def __init__(self, rt: Runtime, model_name: str = "mapanything_apache") -> None:
        self.spec = gate(rt, model_name)
        torch = require("torch", purpose="MapAnything inference")
        module = require("mapanything.models", purpose="MapAnything inference")
        model_cls = getattr(module, "MapAnything", None)
        if model_cls is None:
            raise RuntimeError("installed mapanything package does not expose models.MapAnything")
        self._torch = torch
        self.device = torch_device(rt)
        self.model = model_cls.from_pretrained(str(checkpoint_path(rt, model_name)), local_files_only=True).to(self.device).eval()

    def predict(
        self,
        image_paths: Sequence[Path],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
    ) -> list[GeometryPrediction]:
        if not image_paths:
            return []
        utils = require("mapanything.utils.image", purpose="MapAnything image preprocessing")
        from PIL import Image
        inputs = []
        for i, path in enumerate(image_paths):
            with Image.open(path) as image:
                view = {"img": np.array(image.convert("RGB"))}
            if intrinsics is not None:
                view["intrinsics"] = intrinsics[i]
            if extrinsics is not None:
                view["camera_poses"] = camera_to_world(extrinsics[i])
                view["is_metric_scale"] = self._torch.tensor([True], device=self.device)
            inputs.append(view)
        views = utils.preprocess_inputs(inputs)
        with self._torch.no_grad():
            outputs = self.model.infer(
                views,
                memory_efficient_inference=True,
                minibatch_size=1,
                use_amp=self.device != "cpu",
                amp_dtype="fp16",
                apply_mask=True,
                apply_confidence_mask=False,
                use_multiview_confidence=True,
            )
        predictions: list[GeometryPrediction] = []
        for view, output in zip(views, outputs, strict=True):
            depth = output.get("depth_z")
            if depth is None:
                raise RuntimeError("MapAnything output contains no camera-Z depth")
            depth = as_numpy(depth).squeeze().astype(np.float32)
            if output.get("mask") is not None:
                depth[~as_numpy(output["mask"]).squeeze().astype(bool)] = 0.0
            predictions.append(GeometryPrediction(
                depth=depth,
                confidence=as_numpy(output["conf"]).squeeze().astype(np.float32)
                if output.get("conf") is not None else None,
                points_world=as_numpy(output["pts3d"]).squeeze()
                if output.get("pts3d") is not None else None,
                pose_world_from_camera=as_numpy(output["camera_poses"]).squeeze()
                if output.get("camera_poses") is not None else None,
                intrinsics=as_numpy(view["intrinsics"]).squeeze()
                if view.get("intrinsics") is not None else as_numpy(output["intrinsics"]).squeeze(),
                metric=True,
            ))
        return predictions
