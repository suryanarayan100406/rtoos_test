#!/usr/bin/env python
"""Assemble a DRISHTI-ready input (real drone VIDEO + GPS SRT) from either:

  * a REAL, openly-licensed aerial image set with per-image GPS EXIF   (``--dataset NAME``)
  * a SYNTHETIC procedural city flyover with GROUND-TRUTH GPS/poses    (``--synthetic``)

Why this exists
---------------
DRISHTI ingests a *video file* + a telemetry track (see ``src/drishti/stages/s0_ingest.py``); it ships
**no** bundled data. Real drone footage that is (a) openly licensed and (b) carries per-frame GPS is
scarce, whereas open photogrammetry *image sets* with real GPS EXIF are plentiful. So for the "real"
path we fetch such a set and honestly repackage its **real pixels + real GPS** into the ``MP4 + SRT``
contract the pipeline expects (only the playback timing is synthesized; the imagery and coordinates are
the source's own). The "synthetic" path fabricates nothing about the real world — it renders a labelled
toy city whose geometry and camera path are known, which is exactly what makes its accuracy *checkable*.

Neither output is committed (``data/`` is git-ignored). Nothing here fabricates measurements that are
presented as real: synthetic data is labelled synthetic in its descriptor name and PROVENANCE.txt.

Usage
-----
    python scripts/make_sample_dataset.py --synthetic                    # no network; runs anywhere
    python scripts/make_sample_dataset.py --dataset aukerman             # real buildings (~543 MB)
    python scripts/make_sample_dataset.py --dataset brighton_beach       # small/fast real set (~62 MB)
    python scripts/make_sample_dataset.py --list                         # show the real registry

Then:
    drishti run --dataset configs/datasets/<name>.yaml --profile max
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------------------
# Real, openly-licensed aerial image sets (OpenDroneMap community example datasets).
# All entries have GPS in EXIF (verified against the ODMdata table). Terms vary per source repo; these
# are the community-standard evaluation datasets — check the individual repo before *redistributing*.
# We fetch the GitHub source tarball and recursively find the JPGs, so the in-repo folder layout and
# default branch name do not matter.
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RealSet:
    repo: str          # owner/name on GitHub
    approx_mb: int
    note: str


REGISTRY: dict[str, RealSet] = {
    "aukerman":       RealSet("OpenDroneMap/odm_data_aukerman", 543, "Farmstead: house + barn + fields (real BUILDINGS)."),
    "brighton_beach": RealSet("pierotofy/drone_dataset_brighton_beach", 62, "Beachfront, 18 imgs - smallest georeferenced set (fast)."),
    "caliterra":      RealSet("OpenDroneMap/odm_data_caliterra", 272, "Rolling terrain + a few structures."),
    "lewis":          RealSet("OpenDroneMap/odm_data_lewis", 610, "Larger survey, 145 imgs."),
    "waterbury":      RealSet("OpenDroneMap/odm_data_waterbury", 1872, "Town (Waterbury, VT): residential + commercial BUILDINGS, 248 geotagged imgs -> ~124s @2fps. Heaviest set (~1.9 GB)."),
}


def _require(mod: str, pip_name: str | None = None):
    try:
        return __import__(mod)
    except ImportError as e:
        raise SystemExit(
            f"[make_sample_dataset] missing dependency '{mod}'. Install it with:\n"
            f"    pip install {pip_name or mod}\n"
            f"(OpenCV is also what the pipeline uses to decode/write frames.)"
        ) from e


# ==================================================================================================
# Shared: write the MP4 + SRT + descriptor from a list of per-frame samples.
# ==================================================================================================
@dataclass
class FrameSample:
    lat: float
    lon: float
    abs_alt: float
    rel_alt: float | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None


def _srt_timecode(t: float) -> str:
    if t < 0:
        t = 0.0
    hh = int(t // 3600)
    mm = int((t % 3600) // 60)
    ss = int(t % 60)
    ms = int(round((t - math.floor(t)) * 1000))
    if ms == 1000:  # rounding spillover
        ss += 1
        ms = 0
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def write_srt(samples: list[FrameSample], fps: float, path: Path) -> None:
    """Emit a DJI-style .SRT the parser in io/telemetry/dji_srt.py accepts (bracketed layout)."""
    dt = 1.0 / fps
    lines: list[str] = []
    for i, s in enumerate(samples):
        start, end = i * dt, (i + 1) * dt
        fields = [f"[latitude: {s.lat:.8f}]", f"[longitude: {s.lon:.8f}]"]
        rel = f"rel_alt: {s.rel_alt:.1f} " if s.rel_alt is not None else ""
        fields.append(f"[{rel}abs_alt: {s.abs_alt:.1f}]")
        for name, val in (("yaw", s.yaw), ("pitch", s.pitch), ("roll", s.roll)):
            if val is not None:
                fields.append(f"[{name}: {val:.1f}]")
        lines += [str(i + 1), f"{_srt_timecode(start)} --> {_srt_timecode(end)}", " ".join(fields), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_descriptor(name: str, video: Path, srt: Path, cfg_dir: Path, synthetic: bool) -> Path:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out = cfg_dir / f"{name}.yaml"
    tag = "SYNTHETIC — labelled toy scene, ground-truth GPS" if synthetic else "REAL open aerial set, real GPS EXIF"
    rel_video = video.relative_to(ROOT).as_posix()
    rel_srt = srt.relative_to(ROOT).as_posix()
    out.write_text(
        f"# {name} — generated by scripts/make_sample_dataset.py ({tag}).\n"
        f"# Inputs live under data/ (git-ignored). Regenerate with that script; do not hand-edit paths.\n"
        f"name: {name}\n"
        f"video: {rel_video}\n"
        f"telemetry:\n"
        f"  format: dji_srt\n"
        f"  path: {rel_srt}\n"
        f"optional:\n"
        f"  imu: null\n"
        f"  baro: null\n"
        f"  intrinsics: null\n"
        f"  rtk: null\n"
        f"crs:\n"
        f"  mode: derive_from_gps\n"
        f"  epsg: null\n"
        f"report:\n"
        f"  check_points: null\n",
        encoding="utf-8",
    )
    return out


def encode_video(frames_iter, size: tuple[int, int], fps: float, path: Path) -> int:
    cv2 = _require("cv2", "opencv-python-headless")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise SystemExit(f"[make_sample_dataset] OpenCV could not open a writer for {path}")
    n = 0
    for frame in frames_iter:
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
        writer.write(frame)
        n += 1
    writer.release()
    return n


def finalize(name: str, samples: list[FrameSample], frames, size, fps: float, synthetic: bool,
             provenance: str) -> None:
    out_dir = ROOT / "data" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / "flight.MP4"
    srt = out_dir / "flight.SRT"

    n_written = encode_video(frames, size, fps, video)
    if n_written != len(samples):
        # keep SRT aligned to what actually landed in the video
        samples = samples[:n_written]
    write_srt(samples, fps, srt)
    desc = write_descriptor(name, video, srt, ROOT / "configs" / "datasets", synthetic)
    (out_dir / "PROVENANCE.txt").write_text(provenance, encoding="utf-8")

    print(f"\nOK  Wrote {n_written} frames @ {fps:g} fps  ->  {video.relative_to(ROOT).as_posix()}")
    print(f"OK  Wrote GPS track ({len(samples)} samples)   ->  {srt.relative_to(ROOT).as_posix()}")
    print(f"OK  Wrote descriptor                          ->  {desc.relative_to(ROOT).as_posix()}")
    print(f"\nNext:\n    drishti run --dataset {desc.relative_to(ROOT).as_posix()} --profile max")
    print(f"    (light check now: drishti run --dataset {desc.relative_to(ROOT).as_posix()} --upto s0_ingest)")


# ==================================================================================================
# REAL path — fetch an open image set, read its GPS EXIF, assemble MP4 + SRT.
# ==================================================================================================
def _dms_to_deg(dms, ref) -> float:
    d, m, s = (float(x) for x in dms)
    val = d + m / 60.0 + s / 3600.0
    return -val if str(ref).upper() in ("S", "W") else val


def _read_gps(path: Path):
    """Return (lat, lon, alt, order_key) from a JPEG's EXIF, or None if it has no usable GPS.

    Handles both Pillow APIs: the modern ``getexif().get_ifd(GPSInfo)`` and the legacy
    ``_getexif()[34853]`` nested dict — drones vary in which one exposes the GPS block.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise SystemExit("[make_sample_dataset] missing 'pillow'. Install: pip install pillow") from e
    try:
        im = Image.open(path)
    except Exception:
        return None

    gps = {}
    order = None
    try:
        exif = im.getexif()
        order = exif.get(36867) or exif.get(306)
        g = exif.get_ifd(0x8825)
        if g:
            gps = dict(g)
    except Exception:
        pass
    if 2 not in gps or 4 not in gps:  # legacy fallback
        try:
            old = im._getexif() or {}
            order = order or old.get(36867) or old.get(306)
            g = old.get(34853)
            if g:
                gps = dict(g)
        except Exception:
            pass
    if 2 not in gps or 4 not in gps:
        return None
    try:
        lat = _dms_to_deg(gps[2], gps.get(1, "N"))
        lon = _dms_to_deg(gps[4], gps.get(3, "E"))
        alt = float(gps.get(6, 0.0))
        alt_ref = gps.get(5, 0)
        if isinstance(alt_ref, bytes):        # DJI stores GPSAltitudeRef as a single byte (0=above,1=below)
            alt_ref = alt_ref[0] if alt_ref else 0
        if int(alt_ref) == 1:
            alt = -alt
    except Exception:
        return None
    return lat, lon, alt, str(order or path.name)  # order: DateTimeOriginal / DateTime / filename


def _exif_diag(path: Path) -> str:
    """Best-effort dump of what EXIF a sample image *does* carry, to explain a no-GPS failure."""
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS, TAGS

        im = Image.open(path)
        exif = im.getexif()
        top = sorted(TAGS.get(k, hex(k)) for k in dict(exif))
        try:
            gps_ifd = {GPSTAGS.get(k, k): v for k, v in dict(exif.get_ifd(0x8825)).items()}
        except Exception:
            gps_ifd = {}
        try:
            legacy = {GPSTAGS.get(k, k): v for k, v in dict((im._getexif() or {}).get(34853) or {}).items()}
        except Exception:
            legacy = {}
        return (
            f"    sample image      : {path.name}\n"
            f"    top-level EXIF    : {top[:12]}\n"
            f"    GPS IFD (modern)  : {gps_ifd or '(none)'}\n"
            f"    GPS IFD (legacy)  : {legacy or '(none)'}\n"
        )
    except Exception as e:  # noqa: BLE001 - a diagnostic must never mask the real error
        return f"    (could not introspect EXIF: {e})\n"


def fetch_real(name: str, max_width: int, fps: float) -> None:
    rs = REGISTRY[name]
    cv2 = _require("cv2", "opencv-python-headless")
    _require("PIL", "pillow")

    # Persistent cache: iterating on a dataset must not re-download / re-extract each run.
    cache = ROOT / "data" / ".cache" / rs.repo.replace("/", "__")
    tree = cache / "tree"
    cache.mkdir(parents=True, exist_ok=True)

    def _find_jpgs(root: Path) -> list[Path]:
        return sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg")) if root.exists() else []

    jpgs = _find_jpgs(tree)
    if jpgs:
        print(f"using cached extract: {tree}  ({len(jpgs)} images)")
    else:
        tar_path = cache / "src.tar.gz"
        if not (tar_path.exists() and tar_path.stat().st_size > 0):
            got = False
            for branch in ("master", "main"):
                url = f"https://github.com/{rs.repo}/archive/refs/heads/{branch}.tar.gz"
                try:
                    print(f"downloading {rs.repo}@{branch} (~{rs.approx_mb} MB) ...")
                    _download(url, tar_path)
                    got = True
                    break
                except Exception as e:  # noqa: BLE001 - try the other branch name
                    print(f"  {branch}: {e}")
                    tar_path.unlink(missing_ok=True)
            if not got:
                raise SystemExit(f"[make_sample_dataset] could not download {rs.repo} (tried master/main).")

        print("... extracting")
        staging = cache / "_staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        with tarfile.open(tar_path) as tf:
            tf.extractall(staging)  # noqa: S202 - trusted GitHub source tarball
        shutil.rmtree(tree, ignore_errors=True)
        staging.replace(tree)             # marks the extraction complete (atomic within the cache dir)
        tar_path.unlink(missing_ok=True)  # keep the cache lean; the extracted tree is what we reuse
        jpgs = _find_jpgs(tree)

    if not jpgs:
        raise SystemExit(f"[make_sample_dataset] no JPEGs found in {rs.repo}.")
    print(f"... reading GPS EXIF from {len(jpgs)} images")

    tagged = []
    for p in jpgs:
        g = _read_gps(p)
        if g:
            tagged.append((g[3], p, g[0], g[1], g[2]))
    if not tagged:
        raise SystemExit(
            f"[make_sample_dataset] '{name}': found {len(jpgs)} images but none carried GPS EXIF - "
            f"cannot georeference honestly. Diagnostics for the first image:\n"
            + _exif_diag(jpgs[0])
            + "  Pick a dataset known to carry GPS (see --list)."
        )
    tagged.sort(key=lambda x: x[0])  # flight order by capture time (fallback: filename)
    if len(tagged) < len(jpgs):
        print(f"  ({len(jpgs) - len(tagged)} images skipped: no GPS EXIF)")

    # stream frames to the encoder, downscaling to max_width; collect matching samples
    first = cv2.imread(str(tagged[0][1]))
    if first is None:
        raise SystemExit(f"[make_sample_dataset] OpenCV failed to read {tagged[0][1]}")
    h, w = first.shape[:2]
    scale = min(1.0, max_width / float(w))
    size = (int(round(w * scale)), int(round(h * scale)))

    samples = [FrameSample(lat=lat, lon=lon, abs_alt=alt) for (_o, _p, lat, lon, alt) in tagged]

    def frames():
        for _o, p, *_ in tagged:
            img = cv2.imread(str(p))
            if img is not None:
                yield img

    provenance = (
        f"REAL open aerial dataset assembled by scripts/make_sample_dataset.py\n"
        f"Source repo : https://github.com/{rs.repo}\n"
        f"Note        : {rs.note}\n"
        f"Images used : {len(tagged)} JPEGs carrying GPS EXIF (of {len(jpgs)} found)\n"
        f"Assembly    : real pixels + real per-image GPS, muxed into MP4 @ {fps:g} fps + DJI-style SRT.\n"
        f"              Only playback timing is synthesized; imagery & coordinates are the source's own.\n"
        f"Terms       : OpenDroneMap community example dataset - verify the source repo before redistributing.\n"
    )
    finalize(name, samples, frames(), size, fps, synthetic=False, provenance=provenance)


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "drishti-sample-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed https GitHub host
        if getattr(r, "status", 200) >= 400:
            raise RuntimeError(f"HTTP {r.status}")
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB ({pct:4.1f}%)", end="", flush=True)
        if total:
            print()


# ==================================================================================================
# SYNTHETIC path — render a labelled toy city with known geometry + ground-truth GPS/poses.
# Real multi-view parallax (so SfM/MVS actually reconstruct it); nothing is claimed to be real-world.
# ==================================================================================================
def make_synthetic(name: str, n_frames: int, seed: int, size: tuple[int, int], fps: float) -> None:
    import random as _random

    np = _require("numpy")
    cv2 = _require("cv2", "opencv-python-headless")
    rng = _random.Random(seed)
    npr = np.random.default_rng(seed)

    W, H = size
    fov_x = math.radians(70.0)
    fx = (W / 2) / math.tan(fov_x / 2)  # pinhole focal; project() uses fx, W/2, H/2 as the intrinsics

    # --- scene origin (arbitrary; labelled synthetic). Bengaluru-ish so UTM zone is realistic. ---
    lat0, lon0, ground_alt = 12.97160, 77.59460, 900.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    # --- buildings: boxes on a gridded block plan with street gaps ---
    buildings = []
    for gx in range(-2, 3):
        for gy in range(-2, 3):
            if rng.random() < 0.22:
                continue  # a gap => street/plaza
            cx = gx * 22 + rng.uniform(-2, 2)
            cy = gy * 22 + rng.uniform(-2, 2)
            w, d = rng.uniform(10, 16), rng.uniform(10, 16)
            ht = rng.uniform(8, 30)
            base = np.array([rng.uniform(90, 170)] * 3) + npr.uniform(-25, 25, 3)
            buildings.append((cx, cy, w, d, ht, np.clip(base, 40, 220)))

    def box_faces(cx, cy, w, d, ht, base):
        x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - d / 2, cy + d / 2
        v = {
            "roof": ([(x0, y0, ht), (x1, y0, ht), (x1, y1, ht), (x0, y1, ht)], (0, 0, 1)),
            "xp":   ([(x1, y0, 0), (x1, y1, 0), (x1, y1, ht), (x1, y0, ht)], (1, 0, 0)),
            "xn":   ([(x0, y0, 0), (x0, y1, 0), (x0, y1, ht), (x0, y0, ht)], (-1, 0, 0)),
            "yp":   ([(x0, y1, 0), (x1, y1, 0), (x1, y1, ht), (x0, y1, ht)], (0, 1, 0)),
            "yn":   ([(x0, y0, 0), (x1, y0, 0), (x1, y0, ht), (x0, y0, ht)], (0, -1, 0)),
        }
        out = []
        for kind, (pts, nrm) in v.items():
            col = base * (1.12 if kind == "roof" else 1.0)
            out.append((np.array(pts, float), np.array(nrm, float), col))
        return out

    faces = [f for b in buildings for f in box_faces(*b)]

    # --- consistent trackable features (fixed 3D points, fixed colors) so SfM has correspondences ---
    ground_pts = npr.uniform(-58, 58, (1600, 2))
    ground_xyz = np.column_stack([ground_pts, np.zeros(len(ground_pts))])
    ground_col = npr.integers(60, 150, (len(ground_pts), 3)).astype(float)
    face_pts_xyz, face_pts_col = [], []
    for verts, _n, _c in faces:
        p0, p1, p3 = verts[0], verts[1], verts[3]
        for _ in range(10):
            a, b = npr.random(), npr.random()
            face_pts_xyz.append(p0 + a * (p1 - p0) + b * (p3 - p0))
        face_pts_col.append(npr.integers(30, 220, (10, 3)).astype(float))
    face_pts_xyz = np.array(face_pts_xyz) if face_pts_xyz else np.zeros((0, 3))
    face_pts_col = np.vstack(face_pts_col) if face_pts_col else np.zeros((0, 3))

    light = np.array([0.3, 0.4, 1.0])
    light /= np.linalg.norm(light)

    # --- flight path: serpentine (boustrophedon) at altitude, heading follows motion ---
    alt_agl = 72.0
    wps = [(-38, -46), (-38, 46), (0, 46), (0, -46), (38, -46), (38, 46)]
    wps = [np.array([x, y, alt_agl], float) for x, y in wps]
    seglen = [np.linalg.norm(wps[i + 1] - wps[i]) for i in range(len(wps) - 1)]
    total = sum(seglen)
    cams = []
    for i in range(n_frames):
        s = total * i / max(1, n_frames - 1)
        acc = 0.0
        for j, L in enumerate(seglen):
            if s <= acc + L or j == len(seglen) - 1:
                u = (s - acc) / L if L else 0.0
                pos = wps[j] + u * (wps[j + 1] - wps[j])
                heading = (wps[j + 1] - wps[j])[:2]
                cams.append((pos, heading))
                break
            acc += L

    def look_at(eye, target, up=(0, 0, 1)):
        z = target - eye
        z /= np.linalg.norm(z)
        ref = np.array(up, float)
        if abs(float(z @ ref)) > 0.999:
            ref = np.array([0.0, 1.0, 0.0])
        x = np.cross(ref, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return np.stack([x, y, z])  # world -> camera (rows)

    def project(P, R, eye):
        pc = (R @ (P - eye).T).T
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * pc[:, 0] / z + W / 2
            v = fx * pc[:, 1] / z + H / 2
        return np.column_stack([u, v]), z

    samples: list[FrameSample] = []
    rendered = []
    for pos, heading in cams:
        tilt = 0.22  # ~13° off nadir => oblique parallax, facades visible
        hn = heading / (np.linalg.norm(heading) + 1e-9)
        target = np.array([pos[0] + hn[0] * pos[2] * tilt, pos[1] + hn[1] * pos[2] * tilt, 0.0])
        R = look_at(pos, target)

        img = np.full((H, W, 3), (70, 65, 60), np.uint8)  # ground base (BGR)

        gp, gz = project(ground_xyz, R, pos)
        for (u, v), zz, col in zip(gp, gz, ground_col, strict=True):
            if zz > 1 and -5 < u < W + 5 and -5 < v < H + 5:
                cv2.circle(img, (int(u), int(v)), 1, tuple(int(c) for c in col), -1)

        drawn = []
        for verts, nrm, col in faces:
            pc = (R @ (verts - pos).T).T
            if np.any(pc[:, 2] <= 1):
                continue
            uv, _z = project(verts, R, pos)
            cen = uv.mean(0)
            if not (-2 * W < cen[0] < 3 * W and -2 * H < cen[1] < 3 * H):
                continue
            shade = 0.45 + 0.55 * max(0.0, float(nrm @ light))
            bgr = tuple(int(np.clip(c * shade, 0, 255)) for c in col)
            drawn.append((float(pc[:, 2].mean()), uv.astype(np.int32), bgr))
        for _depth, poly, bgr in sorted(drawn, key=lambda t: -t[0]):  # painter: far -> near
            cv2.fillConvexPoly(img, poly, bgr)
            cv2.polylines(img, [poly], True, (20, 20, 20), 1, cv2.LINE_AA)

        if len(face_pts_xyz):
            fp, fz = project(face_pts_xyz, R, pos)
            for (u, v), zz, col in zip(fp, fz, face_pts_col, strict=True):
                if zz > 1 and 0 <= u < W and 0 <= v < H:
                    cv2.circle(img, (int(u), int(v)), 1, tuple(int(c) for c in col), -1)

        rendered.append(img)
        yaw = (math.degrees(math.atan2(hn[0], hn[1]))) % 360.0  # 0=N, clockwise
        samples.append(FrameSample(
            lat=lat0 + pos[1] / m_per_deg_lat,
            lon=lon0 + pos[0] / m_per_deg_lon,
            abs_alt=ground_alt + pos[2],
            rel_alt=pos[2],
            yaw=round(yaw, 1),
            pitch=round(-90 + math.degrees(tilt), 1),
            roll=0.0,
        ))

    provenance = (
        "SYNTHETIC dataset rendered by scripts/make_sample_dataset.py — NOT real-world imagery.\n"
        f"Scene   : procedural city, {len(buildings)} textured buildings on a block plan.\n"
        f"Camera  : serpentine flight, {n_frames} frames @ {alt_agl:g} m AGL, ~13 deg off-nadir.\n"
        f"Origin  : lat {lat0}, lon {lon0} (arbitrary; picked so the derived UTM zone is realistic).\n"
        "GPS/pose: GROUND TRUTH (computed from the known camera path) — usable to VALIDATE accuracy.\n"
        "Purpose : offline smoke test of the full pipeline + a reconstructable scene with real parallax.\n"
    )
    finalize(name, samples, iter(rendered), size, fps, synthetic=True, provenance=provenance)


# ==================================================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", choices=sorted(REGISTRY), help="fetch a REAL open aerial set by name")
    g.add_argument("--synthetic", action="store_true", help="render a synthetic city (no network)")
    g.add_argument("--list", action="store_true", help="list the real dataset registry and exit")
    ap.add_argument("--name", help="output dataset name (default: the dataset name, or 'synthetic_city')")
    ap.add_argument("--fps", type=float, default=0.0, help="video fps (default 2 for real, 6 for synthetic)")
    ap.add_argument("--max-width", type=int, default=1600, help="downscale real frames to this width (px)")
    ap.add_argument("--frames", type=int, default=54, help="synthetic: number of frames")
    ap.add_argument("--width", type=int, default=1280, help="synthetic: frame width")
    ap.add_argument("--height", type=int, default=960, help="synthetic: frame height")
    ap.add_argument("--seed", type=int, default=7, help="synthetic: RNG seed")
    args = ap.parse_args(argv)

    if args.list:
        print("Real, openly-licensed aerial sets (all have GPS EXIF):\n")
        for k, v in REGISTRY.items():
            print(f"  {k:16} ~{v.approx_mb:>4} MB   {v.note}")
        print("\nFetch one with:  python scripts/make_sample_dataset.py --dataset <name>")
        return 0

    if args.synthetic:
        make_synthetic(
            name=args.name or "synthetic_city",
            n_frames=max(8, args.frames),
            seed=args.seed,
            size=(args.width, args.height),
            fps=args.fps or 6.0,
        )
    else:
        fetch_real(name=args.name or args.dataset, max_width=args.max_width, fps=args.fps or 2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
