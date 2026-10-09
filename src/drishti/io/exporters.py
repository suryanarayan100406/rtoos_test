"""Format exporters (real, each guarded by its own dependency).

Converts the canonical bundle products (mesh.ply, dense.ply, GeoTIFFs) into the deliverable formats.
Every exporter either produces a real file or reports precisely why it could not (missing library or
tool) — it never writes a placeholder (AGENTS.md §5). Returns (ok, path_or_reason).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..runtime.deps import available


def validate_export(fmt: str, path: Path) -> tuple[bool, str]:
    """Re-open an exported artifact and validate its structural contract."""
    path = Path(path)
    target = path
    if fmt == "3dtiles":
        target = path / "tileset.json"
    elif fmt == "potree":
        target = path / "metadata.json"
    if fmt == "geotiff" and path.is_dir():
        targets = sorted(path.glob("*.tif"))
        if not targets:
            return False, f"no GeoTIFF files under {path}"
        for tif in targets:
            ok, reason = validate_export("geotiff", tif)
            if not ok:
                return ok, reason
        return True, str(path)
    if not target.is_file() or target.stat().st_size == 0:
        return False, f"export missing or empty after write: {target}"
    try:
        if fmt in {"obj", "ply", "gltf", "glb"}:
            if not available("trimesh"):
                return False, "trimesh unavailable for post-write validation"
            import trimesh

            loaded = trimesh.load(str(target), process=False)
            meshes: list[Any] = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
            if not meshes or not all(len(m.vertices) and __import__("numpy").isfinite(m.vertices).all() for m in meshes):
                return False, f"trimesh reopened no object from {target}"
        elif fmt in {"las", "laz"}:
            if not available("laspy"):
                return False, "laspy unavailable for post-write validation"
            import laspy

            with laspy.open(str(target)) as reader:
                if reader.header.point_count == 0 or reader.header.parse_crs() is None:
                    return False, f"point export has zero points or lacks CRS: {target}"
                count = sum(len(chunk) for chunk in reader.chunk_iterator(250_000))
                if count != reader.header.point_count:
                    return False, f"point export count mismatch: {target}"
        elif fmt == "geotiff":
            if not available("rasterio"):
                return False, "rasterio unavailable for post-write validation"
            import rasterio

            tifs = sorted(path.glob("*.tif")) if path.is_dir() else [path]
            if not tifs:
                return False, f"no GeoTIFF files under {path}"
            with rasterio.open(tifs[0]) as ds:
                if ds.width <= 0 or ds.height <= 0 or ds.count <= 0 or ds.crs is None:
                    return False, f"invalid GeoTIFF metadata in {tifs[0]}"
                for _, window in ds.block_windows(1):
                    ds.read(window=window)  # verify body decompression in bounded windows
        elif fmt in {"3dtiles", "potree"}:
            json.loads(target.read_text(encoding="utf-8"))
        elif fmt == "fbx":
            blender = shutil.which("blender")
            if not blender:
                return False, "Blender unavailable for FBX re-import validation"
            expr = f"import bpy; bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.fbx(filepath={str(target)!r}); assert any(o.type == 'MESH' for o in bpy.data.objects)"
            proc = subprocess.run([blender, "--background", "--python-exit-code", "1", "--python-expr", expr], capture_output=True, text=True)
            if proc.returncode:
                return False, f"FBX re-import failed: {proc.stderr[-300:]}"
    except Exception as exc:
        return False, f"post-write validation failed for {fmt}: {exc}"
    return True, str(path)


def export_obj(mesh_ply: Path, dest: Path, origin_crs=None) -> tuple[bool, str]:
    if not available("trimesh"):
        return False, "trimesh not installed (pip install '.[recon]')"
    import trimesh
    m = trimesh.load(str(mesh_ply), process=False, force="mesh")
    if origin_crs is not None:
        m.apply_translation(origin_crs)
    dest.parent.mkdir(parents=True, exist_ok=True)
    m.export(str(dest))
    return True, str(dest)


def export_glb(mesh_ply: Path, dest: Path, origin_crs=None) -> tuple[bool, str]:
    if not available("trimesh"):
        return False, "trimesh not installed (pip install '.[recon]')"
    import trimesh
    m = trimesh.load(str(mesh_ply), process=False, force="mesh")
    if origin_crs is None:
        origin_crs = m.bounds.mean(axis=0).tolist()
        m.apply_translation(-__import__("numpy").asarray(origin_crs))
    m.metadata.update(origin_crs=origin_crs, coordinate_frame="local_crs_translation")
    dest.parent.mkdir(parents=True, exist_ok=True)
    m.export(str(dest))  # trimesh picks glTF/GLB from extension
    dest.with_suffix(dest.suffix + ".georef.json").write_text(json.dumps({"origin_crs": origin_crs, "coordinate_frame": "local_crs_translation"}), encoding="utf-8")
    return True, str(dest)


def export_las_from_ply(cloud: Path, dest: Path, epsg: int, point_format="las_1_4", *, laz=False):
    """Stream double-precision metric coordinates into LAS/LAZ with explicit CRS and offsets."""
    if not available("laspy") or not available("pyproj"):
        return False, "LAS export needs laspy[lazrs] and pyproj"
    import laspy
    import numpy as np
    from pyproj import CRS

    from .pointcloud import iter_ply_chunks

    header = laspy.LasHeader(point_format=7 if point_format == "las_1_4" else 3, version="1.4")
    header.add_crs(CRS.from_epsg(epsg))
    header.scales = np.full(3, 0.001)
    for pts, _ in iter_ply_chunks(cloud, chunk_points=1):
        header.offsets = np.floor(pts[0]).astype(float)
        break
    tmp = dest.with_name(dest.stem + ".tmp" + dest.suffix)
    try:
        with laspy.open(tmp, mode="w", header=header, do_compress=laz) as writer:
            for points, colors in iter_ply_chunks(cloud, chunk_points=250_000):
                record = laspy.ScaleAwarePointRecord.zeros(len(points), header=header)
                record.x, record.y, record.z = points.T
                if colors is not None:
                    color = colors.astype(np.uint16) * 257  # exact 8-bit → full 16-bit endpoints
                    record.red, record.green, record.blue = color.T
                writer.write_points(record)
        ok, reason = validate_export("laz" if laz else "las", tmp)
        if not ok:
            return False, reason
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return True, str(dest)


def export_las(points, colors, dest: Path, epsg: int | None, point_format: str = "las_1_4",
               laz: bool = False) -> tuple[bool, str]:
    if not available("laspy"):
        return False, "laspy not installed (pip install laspy[lazrs])"
    import laspy
    import numpy as np

    header = laspy.LasHeader(point_format=7 if point_format == "las_1_4" else 3, version="1.4")
    if epsg:
        try:
            header.add_crs(__import__("pyproj").CRS.from_epsg(epsg))
        except Exception:
            pass
    las = laspy.LasData(header)
    # The dense cloud is stored float32, so an upcast to float64 here recovers no precision — it only
    # doubles a 100M+-point array (~6 GB) for nothing. Feed float32 straight in; laspy still does the
    # (value - offset) / scale scaling in float64 internally, so the written LAS coordinates are identical.
    pts = np.asarray(points, dtype=np.float32)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    if colors is not None:
        c = np.asarray(colors)
        if c.max() <= 255:
            c = (c.astype(np.uint16) << 8)  # LAS color is 16-bit
        las.red, las.green, las.blue = c[:, 0], c[:, 1], c[:, 2]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if laz and dest.suffix != ".laz":
        dest = dest.with_suffix(".laz")
    las.write(str(dest))
    return True, str(dest)


def copy_product(src: Path, dest: Path) -> tuple[bool, str]:
    if not Path(src).exists():
        return False, f"source missing: {src}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True, str(dest)


def export_fbx_via_blender(mesh_ply: Path, dest: Path) -> tuple[bool, str]:
    """Convert PLY -> FBX out-of-process with Blender (GPL isolation). Skipped if blender absent."""
    blender = shutil.which("blender")
    if not blender:
        return False, "blender not found on PATH (FBX export needs headless Blender)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    importer = (
        f"bpy.ops.import_scene.gltf(filepath={str(mesh_ply)!r});"
        if mesh_ply.suffix.lower() in {".gltf", ".glb"}
        else f"bpy.ops.wm.ply_import(filepath={str(mesh_ply)!r});"
    )
    script = (
        "import bpy,sys;"
        "bpy.ops.wm.read_factory_settings(use_empty=True);"
        + importer
        + f"bpy.ops.export_scene.fbx(filepath={str(dest)!r})"
    )
    proc = subprocess.run([blender, "--background", "--python-exit-code", "1", "--python-expr", script],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not dest.exists():
        return False, f"blender FBX export failed: {proc.stderr[-300:] or proc.stdout[-300:]}"
    return True, str(dest)


def export_3dtiles(mesh_or_ply: Path, dest_dir: Path) -> tuple[bool, str]:
    tool = shutil.which("py3dtiles") or shutil.which("3d-tiles-tools")
    if not tool:
        return False, "no 3D Tiles converter on PATH (py3dtiles / 3d-tiles-tools)"
    dest_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([tool, "convert", str(mesh_or_ply), "--out", str(dest_dir)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return False, f"3D Tiles conversion failed: {proc.stderr[-300:]}"
    return True, str(dest_dir)


def export_potree(ply: Path, dest_dir: Path) -> tuple[bool, str]:
    tool = shutil.which("PotreeConverter")
    if not tool:
        return False, "PotreeConverter not on PATH"
    dest_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([tool, str(ply), "-o", str(dest_dir)], capture_output=True, text=True)
    if proc.returncode != 0:
        return False, f"PotreeConverter failed: {proc.stderr[-300:]}"
    return True, str(dest_dir)
