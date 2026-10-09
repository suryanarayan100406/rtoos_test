"""Bundle lifecycle: create/open round-trip, manifest persistence, input registration."""
import pytest

from drishti.bundle import Bundle
from drishti.bundle.manifest import StageRecord


def test_create_open_roundtrip(tmp_path):
    b = Bundle.create(
        tmp_path, run_id="r1", dataset_name="d",
        config_snapshot={"a": 1}, dataset_snapshot={"name": "d"}, seeds={"global": 7},
    )
    assert (tmp_path / "r1" / "manifest.json").is_file()
    b.manifest.upsert_stage(StageRecord(name="s0_ingest", status="done", outputs_hash="abc"))
    b.save_manifest()

    reopened = Bundle.open(tmp_path / "r1")
    assert reopened.manifest.run_id == "r1"
    assert reopened.manifest.dataset_name == "d"
    assert reopened.manifest.config_snapshot == {"a": 1}
    assert reopened.manifest.seeds == {"global": 7}
    assert reopened.manifest.stage("s0_ingest").status == "done"


def test_open_missing_bundle_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Bundle.open(tmp_path / "nope")


def test_upsert_stage_replaces_not_appends(tmp_path):
    b = Bundle.create(tmp_path)
    b.manifest.upsert_stage(StageRecord(name="s0_ingest", status="running"))
    b.manifest.upsert_stage(StageRecord(name="s0_ingest", status="done"))
    assert len([s for s in b.manifest.stages if s.name == "s0_ingest"]) == 1
    assert b.manifest.stage("s0_ingest").status == "done"


def test_register_input_missing_required_raises(tmp_path):
    b = Bundle.create(tmp_path)
    with pytest.raises(ValueError):
        b.register_input(None, role="video", required=True)
    with pytest.raises(ValueError):
        b.register_input(tmp_path / "nope.mp4", role="video", required=True)


def test_register_input_optional_absent(tmp_path):
    b = Bundle.create(tmp_path)
    ref = b.register_input(None, role="imu", required=False)
    assert ref.present is False
    assert ref.role == "imu"


def test_register_input_hashes_present_file(tmp_path):
    f = tmp_path / "in.bin"
    f.write_bytes(b"hello")
    b = Bundle.create(tmp_path)
    ref = b.register_input(f, role="video", required=True)
    assert ref.present is True
    assert ref.bytes == 5
    assert ref.sha256 and len(ref.sha256) == 64


def test_write_read_json_roundtrip(tmp_path):
    b = Bundle.create(tmp_path)
    rel = b.write_json("s0_ingest/x.json", {"k": 1, "nested": {"v": [1, 2, 3]}})
    assert b.exists(rel)
    assert b.read_json(rel) == {"k": 1, "nested": {"v": [1, 2, 3]}}


def test_relpath_uses_forward_slashes(tmp_path):
    b = Bundle.create(tmp_path, run_id="r1")
    p = b.stage_dir("s0_ingest") / "frames" / "f.jpg"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    assert b.relpath(p) == "s0_ingest/frames/f.jpg"
