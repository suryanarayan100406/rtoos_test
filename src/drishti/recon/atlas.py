"""UV atlas baking from real calibrated images with metric-depth visibility rejection.

Uses MIT xatlas for UV parameterization. Only supported texels are opaque; unobserved surface remains
transparent and is counted explicitly. Face colors are selected by viewing angle, then verified per
texel against source metric depth. This is a best-view atlas, not a claim of MVS seam optimization.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..runtime.deps import require
from .depthfit import project_points


def bake_atlas(vertices_world, faces, views, K, load_view, *, output_dir: Path,
               atlas_size: int, visibility_tolerance_m: float) -> dict:
    xatlas = require("xatlas", purpose="texture UV generation")
    cv2 = require("cv2", purpose="texture projection")
    trimesh = require("trimesh", purpose="textured mesh output")
    from PIL import Image

    vertices = np.asarray(vertices_world, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    origin = vertices.mean(axis=0)
    local = (vertices - origin).astype(np.float32)
    mapping, uv_faces, uv = xatlas.parametrize(local, faces.astype(np.uint32))
    # Original face order is unchanged by parameterization, but output vertices may be duplicated at seams.
    uv_vertices = vertices[mapping]
    tri = uv_vertices[uv_faces]
    centers = tri.mean(axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    best = np.full(len(faces), -1, dtype=np.int32)
    best_score = np.full(len(faces), -np.inf)
    for i, view in enumerate(views):
        center = np.asarray(view["center_crs"], dtype=np.float64)
        delta = center - centers
        distance = np.linalg.norm(delta, axis=1)
        facing = np.einsum("ij,ij->i", normals, delta) / np.maximum(distance, 1e-12)
        pixels, z = project_points(K, np.asarray(view["R"]), center, centers)
        ok = (np.isfinite(pixels).all(axis=1) & (z > 0) & (facing > 0)
              & (pixels[:, 0] >= 0) & (pixels[:, 0] < view["width"])
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < view["height"]))
        # Choose view by actual incidence and resolution (distance), not an arbitrary confidence boost.
        score = facing / np.maximum(distance, 1e-12)
        replace = ok & (score > best_score)
        best[replace], best_score[replace] = i, score[replace]

    rgba = np.zeros((atlas_size, atlas_size, 4), dtype=np.uint8)
    uv_pixels = np.column_stack([uv[:, 0] * (atlas_size - 1), (1.0 - uv[:, 1]) * (atlas_size - 1)])
    textured_faces = 0
    for i, view in enumerate(views):
        image, depth = load_view(view)
        for face_index in np.where(best == i)[0]:
            triangle_uv = uv_pixels[uv_faces[face_index]]
            lo = np.maximum(np.floor(triangle_uv.min(axis=0)).astype(int), 0)
            hi = np.minimum(np.ceil(triangle_uv.max(axis=0)).astype(int), atlas_size - 1)
            if np.any(hi < lo):
                continue
            yy, xx = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
            a, b, c = triangle_uv
            det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(det) < 1e-10:
                continue
            wa = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / det
            wb = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / det
            wc = 1 - wa - wb
            inside = (wa >= 0) & (wb >= 0) & (wc >= 0)
            if not inside.any():
                continue
            weights = np.column_stack([wa[inside], wb[inside], wc[inside]])
            world = weights @ tri[face_index]
            pixels, z = project_points(K, np.asarray(view["R"]), np.asarray(view["center_crs"]), world)
            finite = np.isfinite(pixels).all(axis=1) & np.isfinite(z) & (z > 0)
            u = np.where(finite, pixels[:, 0], -1).astype(np.float32)
            v = np.where(finite, pixels[:, 1], -1).astype(np.float32)
            expected = cv2.remap(np.asarray(depth, dtype=np.float32), u[:, None], v[:, None], cv2.INTER_NEAREST).ravel()
            visible = (finite & (expected > 0) & (np.abs(expected - z) <= visibility_tolerance_m)
                       & (u >= 0) & (u < image.shape[1]) & (v >= 0) & (v < image.shape[0]))
            if not visible.any():
                continue
            colors = cv2.remap(image, u[:, None], v[:, None], cv2.INTER_LINEAR)[:, 0, ::-1]
            iy, ix = yy[inside][visible], xx[inside][visible]
            rgba[iy, ix, :3] = colors[visible]
            rgba[iy, ix, 3] = 255
            textured_faces += 1
        del image, depth
    if not textured_faces:
        raise RuntimeError("texture atlas has no camera/depth-supported texels")
    output_dir.mkdir(parents=True, exist_ok=True)
    texture = Image.fromarray(rgba)
    texture.save(output_dir / "texture.png")
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=texture,
                                                   metallicFactor=0.0, roughnessFactor=1.0,
                                                   alphaMode="MASK")
    visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    mesh = trimesh.Trimesh(vertices=uv_vertices - origin, faces=uv_faces, visual=visual, process=False)
    asset = output_dir / "textured.glb"
    mesh.export(asset)
    reopened = trimesh.load(asset, process=False, force="scene")
    if not reopened.geometry or not any(m.visual.kind == "texture" for m in reopened.geometry.values()):
        raise RuntimeError("textured GLB did not reopen with a UV material")
    return {
        "method": "xatlas_best_view_depth_visible", "asset": str(asset),
        "texture_image": str(output_dir / "texture.png"), "origin_crs": origin.tolist(),
        "coordinate_frame": "local_crs_translation", "n_faces": len(faces),
        "n_textured_faces": textured_faces, "n_opaque_texels": int((rgba[..., 3] > 0).sum()),
        "atlas_size": atlas_size, "visibility_tolerance_m": visibility_tolerance_m,
        "unsupported_texels": "transparent", "seam_optimization": False,
        "measurement_authoritative": False,
    }
