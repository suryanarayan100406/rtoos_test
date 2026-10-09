"""Metric 3D Gaussian Splatting backend using the maintained gsplat rasterizer.

This is a deliberately bounded training path: metric depth initializes Gaussian means, gsplat optimizes
appearance/position/scale/opacity against the real keyframes, and rendered depth remains tied to the
metric input.  It writes a Gaussian checkpoint as well as a colored point representation for existing
mesh/geospatial consumers.  Missing CUDA/gsplat fails loudly; it never falls back while claiming that
Gaussian optimization ran.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..runtime.deps import require


@dataclass
class GaussianResult:
    points: np.ndarray
    colors: np.ndarray
    checkpoint: Path
    iterations: int
    final_rgb_loss: float
    final_depth_loss: float | None
    n_gaussians: int
    origin_crs: np.ndarray


def _resize_nearest(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    ys = np.floor(np.arange(height) * arr.shape[0] / height).astype(int)
    xs = np.floor(np.arange(width) * arr.shape[1] / width).astype(int)
    return arr[np.ix_(ys, xs)]


def prepare_metric_view(color_bgr, depth, extr, K, render_h, render_w, origin_crs, max_samples, rng):
    """Build a render view and metric samples in the same recentered local frame, before float32."""
    from .camera import recenter_extrinsic

    color = _resize_nearest(color_bgr, render_h, render_w)[..., ::-1].astype(np.float32) / 255.0
    depth_small = _resize_nearest(depth, render_h, render_w).astype(np.float32)
    depth_small[~np.isfinite(depth_small) | (depth_small <= 0)] = 0
    ext_local = recenter_extrinsic(extr, origin_crs)
    vs, us = np.where(depth_small > 0)
    if len(vs) > max_samples:
        take = rng.choice(len(vs), max_samples, replace=False)
        vs, us = vs[take], us[take]
    d = depth_small[vs, us]
    cam = np.column_stack(((us - K[0, 2]) / K[0, 0] * d,
                           (vs - K[1, 2]) / K[1, 1] * d, d))
    world_local = (cam - ext_local[:3, 3]) @ ext_local[:3, :3]
    return ({"image": color, "depth": depth_small, "extr": ext_local.astype(np.float32), "K": K.copy()},
            world_local.astype(np.float32), color[vs, us])


def train_metric_gaussians(
    frame_specs: Sequence[dict],
    load: Callable[[dict], tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    voxel_m: float,
    iters: int,
    max_gaussians: int,
    view_batch: int,
    checkpoint_every: int,
    output_dir: Path,
    seed: int,
) -> GaussianResult:
    """Initialize and optimize a bounded metric Gaussian scene from real depth/color frames."""
    torch = require("torch", purpose="metric Gaussian optimization")
    gsplat = require("gsplat", purpose="metric Gaussian rasterization")
    if not torch.cuda.is_available():
        raise RuntimeError("gsplat Gaussian reconstruction requires a CUDA device")
    if iters <= 0 or max_gaussians <= 0:
        raise ValueError("Gaussian iterations and max_gaussians must be positive")
    if not frame_specs:
        raise RuntimeError("Gaussian reconstruction received no metric depth frame descriptors")

    from torch import nn

    device = torch.device("cuda")
    render_long_side = min(max(width, height), 768)
    render_scale = render_long_side / max(width, height)
    render_w = max(1, int(round(width * render_scale)))
    render_h = max(1, int(round(height * render_scale)))
    sx, sy = render_w / width, render_h / height
    K = np.array([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]], dtype=np.float32)

    rng = np.random.default_rng(seed)
    frame_data: list[dict[str, np.ndarray]] = []
    initial_points: list[np.ndarray] = []
    initial_colors: list[np.ndarray] = []
    per_frame_cap = max(1, max_gaussians // max(1, len(frame_specs)))
    centers = np.asarray([
        -np.asarray(spec["extr"])[:3, :3].T @ np.asarray(spec["extr"])[:3, 3]
        for spec in frame_specs
    ], dtype=np.float64)
    origin_crs = centers.mean(axis=0)
    for spec in frame_specs:
        loaded = load(spec)
        if loaded is None:
            continue
        color_bgr, depth, extr = loaded
        view, points, colors = prepare_metric_view(
            color_bgr, depth, extr, K, render_h, render_w, origin_crs, per_frame_cap, rng,
        )
        if not len(points):
            continue
        frame_data.append(view)
        initial_points.append(points)
        initial_colors.append(colors)

    if not frame_data or not initial_points:
        raise RuntimeError("Gaussian reconstruction received no valid metric depth frames")
    points = np.concatenate(initial_points, axis=0)
    colors = np.concatenate(initial_colors, axis=0)
    if len(points) > max_gaussians:
        keep = rng.choice(len(points), max_gaussians, replace=False)
        points, colors = points[keep], colors[keep]
    # A metric voxel key removes redundant samples before allocating optimizer state.
    keys = np.floor(points / max(voxel_m, 1e-4)).astype(np.int64)
    _, unique = np.unique(keys, axis=0, return_index=True)
    unique = np.sort(unique)
    points, colors = points[unique], colors[unique]
    if len(points) > max_gaussians:
        points, colors = points[:max_gaussians], colors[:max_gaussians]

    means = nn.Parameter(torch.as_tensor(points, device=device))
    scales = nn.Parameter(torch.full((len(points), 3), np.log(max(voxel_m, 1e-3)), device=device))
    quats = nn.Parameter(torch.zeros((len(points), 4), device=device))
    with torch.no_grad():
        quats[:, 0] = 1.0
    opacity_logits = nn.Parameter(torch.full((len(points),), 2.0, device=device))
    color_logits = nn.Parameter(torch.logit(torch.as_tensor(np.clip(colors, 1e-4, 1 - 1e-4), device=device)))
    optimizer = torch.optim.Adam([means, scales, quats, opacity_logits, color_logits], lr=1e-2)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "gaussians.pt"
    final_rgb_loss = float("nan")
    final_depth_loss: float | None = None
    regularizers = {"depth"}
    for iteration in range(1, iters + 1):
        count = min(max(1, int(view_batch)), len(frame_data))
        selected = [(iteration * count + j) % len(frame_data) for j in range(count)]
        viewmats = torch.as_tensor(np.stack([frame_data[i]["extr"] for i in selected]), device=device)
        Ks = torch.as_tensor(np.stack([frame_data[i]["K"] for i in selected]), device=device)
        target_rgb = torch.as_tensor(np.stack([frame_data[i]["image"] for i in selected]), device=device)
        target_depth = torch.as_tensor(np.stack([frame_data[i]["depth"] for i in selected]), device=device)
        render, alpha, _ = gsplat.rasterization(
            means, torch.nn.functional.normalize(quats, dim=-1), torch.exp(scales),
            torch.sigmoid(opacity_logits), torch.sigmoid(color_logits), viewmats, Ks,
            render_w, render_h, packed=True, render_mode="RGB+ED",
        )
        rgb_loss = torch.nn.functional.l1_loss(render[..., :3], target_rgb)
        valid_depth = (target_depth > 0) & (alpha[..., 0] > 1e-3)
        if valid_depth.any():
            depth_loss = torch.nn.functional.smooth_l1_loss(
                render[..., 3][valid_depth], target_depth[valid_depth]
            )
            loss = rgb_loss + 0.1 * depth_loss
            final_depth_loss = float(depth_loss.detach().cpu())
        else:
            depth_loss = None
            loss = rgb_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_rgb_loss = float(rgb_loss.detach().cpu())
        if checkpoint_every > 0 and (iteration % checkpoint_every == 0 or iteration == iters):
            torch.save({
                "means": means.detach().cpu(), "scales": scales.detach().cpu(),
                "quats": quats.detach().cpu(), "opacities": opacity_logits.detach().cpu(),
                "colors": color_logits.detach().cpu(), "iteration": iteration,
                "render_size": [render_w, render_h], "geometry_regularizers": sorted(regularizers),
                "origin_crs": origin_crs.tolist(),
            }, checkpoint)

    with torch.no_grad():
        keep = torch.sigmoid(opacity_logits).detach().cpu().numpy() > 1e-3
        final_points = means.detach().cpu().numpy()[keep].astype(np.float32)
        final_colors = torch.sigmoid(color_logits).detach().cpu().numpy()[keep].astype(np.float32)
    if len(final_points) == 0:
        raise RuntimeError("Gaussian optimization pruned every Gaussian")
    return GaussianResult(
        points=final_points, colors=final_colors, checkpoint=checkpoint,
        iterations=iters, final_rgb_loss=final_rgb_loss, final_depth_loss=final_depth_loss,
        n_gaussians=len(final_points), origin_crs=origin_crs,
    )
