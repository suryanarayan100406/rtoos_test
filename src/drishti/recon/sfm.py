"""Structure-from-Motion via COLMAP (pycolmap) — real, guarded.

Runs feature extraction, matching and incremental mapping on the selected keyframes to recover camera
intrinsics + poses in an arbitrary "visual" frame, plus a sparse point cloud. No pycolmap installed =>
loud failure (never fabricated poses). The visual frame is metrically anchored later in S2 by a Sim(3)
fit to the GPS track.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..runtime.deps import require


@dataclass
class CameraPose:
    name: str
    R: list[list[float]]      # 3x3 rotation, world -> camera
    t: list[float]            # 3, translation, world -> camera
    center: list[float]       # 3, camera center in the visual world frame


@dataclass
class SfmResult:
    poses: list[CameraPose] = field(default_factory=list)
    points_xyz: list[list[float]] = field(default_factory=list)
    intrinsics: dict = field(default_factory=dict)
    n_registered: int = 0
    n_input_images: int = 0


def run_colmap_sfm(
    image_paths: list[Path],
    work_dir: Path,
    matcher: str = "sequential",
    self_calibrate: bool = True,
) -> SfmResult:
    """Run COLMAP SfM over ``image_paths``; return normalized poses + sparse points."""
    pycolmap = require("pycolmap", purpose="Structure-from-Motion")
    import numpy as np

    work_dir.mkdir(parents=True, exist_ok=True)
    # COLMAP wants all images in one directory; build a symlink/flat view via a shared parent.
    image_dir = image_paths[0].parent
    image_list = [p.name for p in image_paths]
    db_path = work_dir / "database.db"
    out_dir = work_dir / "sparse"
    out_dir.mkdir(exist_ok=True)

    camera_mode = pycolmap.CameraMode.SINGLE
    # The 3rd positional arg is the image-name allowlist. pycolmap renamed this *keyword*
    # (image_list -> image_names) around 0.6; its position is unchanged, so pass it
    # positionally to work on both old and new pycolmap. camera_mode is stable by name.
    pycolmap.extract_features(
        str(db_path), str(image_dir), image_list, camera_mode=camera_mode,
    )
    if matcher == "exhaustive":
        pycolmap.match_exhaustive(str(db_path))
    elif matcher == "vocab_tree":
        pycolmap.match_vocabtree(str(db_path))
    else:
        pycolmap.match_sequential(str(db_path))

    maps = pycolmap.incremental_mapping(str(db_path), str(image_dir), str(out_dir))
    if not maps:
        raise RuntimeError(
            "COLMAP registered no images — insufficient overlap/features. "
            "Try more keyframes (lower min_parallax_px) or exhaustive matching."
        )
    rec = maps[max(maps, key=lambda k: maps[k].num_reg_images())]

    poses: list[CameraPose] = []
    for _img_id, image in rec.images.items():
        # pycolmap's rig refactor turned Image.cam_from_world into a *method*; older
        # versions expose it as a property. Call it when callable so both yield a Rigid3d.
        cfw = image.cam_from_world
        if callable(cfw):
            cfw = cfw()
        R = np.asarray(cfw.rotation.matrix(), dtype=float)
        t = np.asarray(cfw.translation, dtype=float).reshape(3)
        center = (-R.T @ t)
        poses.append(CameraPose(
            name=image.name, R=R.tolist(), t=t.tolist(), center=center.tolist(),
        ))

    sequence = {p.name: i for i, p in enumerate(image_paths)}
    poses.sort(key=lambda p: sequence[p.name])

    points = [list(map(float, p.xyz)) for p in rec.points3D.values()]
    cams = rec.cameras
    if len(cams) != 1:
        raise ValueError("SfM returned multiple cameras; a shared calibration cannot represent this recording")
    intr = {}
    if cams:
        cam = next(iter(cams.values()))
        intr = {"model": str(cam.model.name if hasattr(cam.model, "name") else cam.model),
                "width": int(cam.width), "height": int(cam.height),
                "params": list(map(float, cam.params))}
    return SfmResult(
        poses=poses, points_xyz=points, intrinsics=intr,
        n_registered=len(poses), n_input_images=len(image_paths),
    )
