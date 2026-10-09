"""S1 persists GNSS-track coverage after QA and either keyframe selector."""
from pathlib import Path

import numpy as np
import pytest

from drishti.bundle import Bundle
from drishti.config import DrishtiConfig
from drishti.stages.s1_frameqa import FrameQAStage


@pytest.mark.parametrize("method", ["fixed_stride", "parallax"])
@pytest.mark.parametrize(
    "longitudes,rejected,expected_coverage",
    [
        pytest.param([0.0, 0.001, 0.004], (), [0.0, 0.25, 1.0], id="unequal-segments"),
        pytest.param([0.0], (), [0.0], id="single-frame"),
        pytest.param([0.001, 0.001], (), [0.0, 0.0], id="stationary"),
        pytest.param([None, None], (), [0.0, 0.0], id="no-gps"),
        pytest.param(
            [0.0, 0.001, None, 0.004, 0.006], (),
            [0.0, 0.3333, 0.3333, 0.3333, 1.0], id="gps-gap",
        ),
        pytest.param([0.0, 2.0, 0.001, 0.004], (1,), [0.0, 0.25, 1.0], id="qa-rejected"),
    ],
)
def test_frameqa_writes_coverage(tmp_path, monkeypatch, method, longitudes, rejected, expected_coverage):
    bundle = Bundle.create(tmp_path, run_id="frameqa")
    frames = [
        {"index": i, "t": float(i), "path": f"s0_ingest/frames/{i:06d}.jpg",
         "lat": 0.0 if lon is not None else None, "lon": lon, "alt": 100.0}
        for i, lon in enumerate(longitudes)
    ]
    bundle.write_json("s0_ingest/frames.json", {"frames": frames})
    cfg = DrishtiConfig(frameqa={"keyframe": {"method": method, "stride": 1}})

    # Synthetic pixels are confined to this test. Exercise actual QA scoring and JSON output;
    # isolate image decoding and the native ORB boundary from this stage-contract regression.
    sharp = (64 + 128 * (np.indices((16, 16)).sum(axis=0) % 2)).astype(np.uint8)
    flat = np.full_like(sharp, 128)
    monkeypatch.setattr(
        "drishti.io.video.read_image",
        lambda path: flat if int(Path(path).stem) in rejected else sharp,
    )
    if method == "parallax":
        monkeypatch.setattr("drishti.runtime.deps.require", lambda *args, **kwargs: object())
        monkeypatch.setattr(
            "drishti.stages.s1_frameqa._median_orb_displacement",
            lambda *args: cfg.frameqa.keyframe.min_parallax_px,
        )

    result = FrameQAStage().run(bundle, cfg, None)
    qa = bundle.read_json(result.outputs["frameqa"])
    accepted = [frame for frame in qa["frames"] if frame["accepted"]]
    assert [frame["coverage_score"] for frame in accepted] == pytest.approx(expected_coverage)
    assert qa["n_frames"] == len(frames)
    assert qa["n_accepted"] == len(expected_coverage)
    assert result.metrics["n_keyframes"] == len(expected_coverage)
    assert result.confidence_summary == round(len(accepted) / len(frames), 3)
    assert (result.degraded_reason is not None) == (len(accepted) < 2)
    for i in rejected:
        assert qa["frames"][i]["selection_reason"] == "qa_rejected"
        assert qa["frames"][i]["coverage_score"] == 0.0
    keyframes = bundle.read_json(result.outputs["keyframes"])["keyframes"]
    assert [frame["index"] for frame in keyframes] == [frame["index"] for frame in accepted]
