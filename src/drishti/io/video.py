"""Video decoding and frame extraction (real, via PyAV or OpenCV — guarded).

Decodes a recorded clip deterministically and samples frames at a target fps for downstream stages.
No synthetic frames are ever produced; if no decoder is installed, this fails loudly.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..runtime.deps import available, require


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration_s: float
    n_frames: int


@dataclass
class FrameRef:
    index: int          # index within the sampled set
    src_index: int      # frame index in the source stream
    t: float            # seconds from start
    path: str           # bundle-relative path to the saved frame


def probe_video(path: str | Path) -> VideoInfo:
    path = str(path)
    if available("av"):
        av = require("av")
        with av.open(path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            duration = float(container.duration / 1_000_000) if container.duration else 0.0
            n = stream.frames or 0
            return VideoInfo(path, stream.codec_context.width, stream.codec_context.height, fps, duration, n)
    cv2 = require("cv2", purpose="video probing")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoInfo(path, w, h, fps, (n / fps if fps else 0.0), n)


def iter_frames(
    path: str | Path, target_fps: float = 0.0, max_frames: int = 0
) -> Iterator[tuple[int, float, object]]:
    """Yield (src_index, t_seconds, frame_bgr ndarray), sampled to ~target_fps (0 => native)."""
    require("numpy")  # frames are returned as ndarrays; fail loudly if numpy is absent
    info = probe_video(path)
    src_fps = info.fps or 30.0
    stride = max(1, int(round(src_fps / target_fps))) if target_fps > 0 else 1

    if available("av"):
        av = require("av")
        emitted = 0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for i, frame in enumerate(container.decode(stream)):
                if i % stride != 0:
                    continue
                t = float(frame.pts * stream.time_base) if frame.pts is not None else i / src_fps
                img = frame.to_ndarray(format="bgr24")
                yield i, t, img
                emitted += 1
                if max_frames and emitted >= max_frames:
                    return
        return

    cv2 = require("cv2", purpose="frame extraction")
    cap = cv2.VideoCapture(str(path))
    i = emitted = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            yield i, (i / src_fps), frame
            emitted += 1
            if max_frames and emitted >= max_frames:
                break
        i += 1
    cap.release()


def save_frame(img, dest: str | Path) -> None:
    cv2 = require("cv2", purpose="saving frames")
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest), img):
        raise RuntimeError(f"failed to write frame: {dest}")


def read_image(path: str | Path):
    """Read an image file to a BGR ndarray (real decode; fails loudly without a decoder)."""
    cv2 = require("cv2", purpose="reading frames")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {path}")
    return img
