"""Exercise hosted path selection and recovery without downloads or GPU work."""
import ast
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/drishti_colab_full.ipynb"


def _execute_cell(prefix, namespace, overrides=None):
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    source = next(
        "".join(cell["source"]) for cell in cells
        if cell["cell_type"] == "code" and "".join(cell["source"]).startswith(prefix)
    )
    tree = ast.parse(source)
    # Set the operator-editable fields, leaving the cell's workflow intact.
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in (overrides or {}):
                node.value = ast.Constant(overrides[name])
    exec(compile(ast.fix_missing_locations(tree), str(NOTEBOOK), "exec"), namespace)


@pytest.mark.parametrize(
    "directories,kaggle,colab,root",
    [
        ({"/kaggle/working", "/content"}, True, False, "/kaggle/working"),
        ({"/kaggle/working"}, True, False, "/kaggle/working"),
        ({"/content"}, False, True, "/content"),
    ],
)
def test_host_setup_prefers_kaggle(monkeypatch, directories, kaggle, colab, root):
    monkeypatch.setattr(Path, "is_dir", lambda path: str(path) in directories)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    namespace = {}
    _execute_cell("# Hosted setup:", namespace)
    assert namespace["ON_KAGGLE"] is kaggle
    assert namespace["ON_COLAB"] is colab
    assert namespace["HOST_ROOT"] == Path(root)


@pytest.mark.parametrize("archive", [False, True], ids=["directory", "zip"])
def test_option_c_recovers_bundle_with_original_input_root(tmp_path, monkeypatch, archive):
    from drishti.bundle import Bundle

    original = tmp_path / "old_checkout"
    original.mkdir()
    (original / "flight.mp4").write_bytes(b"test video reference")
    (original / "flight.srt").write_bytes(b"test telemetry reference")
    bundle = Bundle.create(
        original / "runs", run_id="interrupted",
        dataset_snapshot={"video": "flight.mp4", "telemetry": {"path": "flight.srt"}},
    )
    new_checkout = tmp_path / "new_checkout"
    new_checkout.mkdir()
    monkeypatch.chdir(new_checkout)
    source = (shutil.make_archive(str(tmp_path / "saved"), "zip", bundle.path.parent, bundle.path.name)
              if archive else str(bundle.path))
    namespace = {"REPO_DIR": new_checkout, "HOST_ROOT": tmp_path, "ON_COLAB": False, "os": os}
    _execute_cell("# == IMPORT · Option C", namespace, {
        "RUN_OPTION_C": True, "BUNDLE_SOURCE": source, "INPUT_ROOT": str(original),
    })
    restored = Path(namespace["RESUME_BUNDLE"])
    assert (restored / "manifest.json").read_bytes() == (bundle.path / "manifest.json").read_bytes()
    assert namespace["BUNDLE"] == str(restored)
    assert namespace["MISSION"] == ""
    assert Path.cwd() == original
    assert restored == bundle.path if not archive else restored != bundle.path


@pytest.mark.parametrize("block,depth", [("6A", "40"), ("6B", "150")])
@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "resume"])
def test_run_cell_preserves_logs_and_uses_selected_interpreter(tmp_path, monkeypatch, block, depth, resume):
    import time

    repo = tmp_path / "code"
    repo.mkdir()
    mission = repo / "mission.yaml"
    mission.write_text("name: test\n")
    bundle = tmp_path / "existing" if resume else repo / "runs" / "test-run"
    log = bundle / "logs/run_console.log"
    log.parent.mkdir(parents=True)
    log.write_text("original failure evidence\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(time, "strftime", lambda fmt: "test-run")
    calls = []

    def launch(cmd, **kwargs):
        calls.append((cmd, Path.cwd()))
        return SimpleNamespace(stdout=io.StringIO("new attempt output\n"), returncode=1, wait=lambda: 1)

    monkeypatch.setattr(subprocess, "Popen", launch)
    selected_python = "/selected/environment/bin/python"
    namespace = {
        "REPO_DIR": repo, "PY": selected_python, "MISSION": str(mission),
        "RESUME_BUNDLE": str(bundle) if resume else None,
    }
    _execute_cell(f"# == RUN · Block {block}", namespace)
    cmd, cwd = calls[0]
    assert cmd[:3] == [selected_python, "-m", "drishti.cli"]
    if resume:
        assert cmd[3:] == ["resume", str(bundle)]
        assert cwd == tmp_path  # retain Option C's chosen input root
    else:
        assert cmd[3] == "run"
        assert cmd[cmd.index("--output-root") + 1] == str(repo / "runs")
        assert f"dense.depth_trunc_m={depth}" in cmd
        assert cwd == repo
    assert namespace["BUNDLE"] == str(bundle)
    assert log.read_text().startswith("original failure evidence\n")
    assert "new attempt output" in log.read_text()


def test_option_e_validates_descriptor_in_target_python(tmp_path, monkeypatch):
    from drishti.config.loader import load_dataset, parse_env_overrides

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    video = tmp_path / "flight.mp4"
    telemetry = tmp_path / "flight.srt"
    video.write_bytes(b"test video reference")
    telemetry.write_bytes(b"test telemetry reference")
    namespace = {"REPO_DIR": tmp_path, "PY": sys.executable, "ON_COLAB": False}
    _execute_cell("# == IMPORT · Option E", namespace, {
        "VIDEO": str(video), "TELEMETRY": str(telemetry), "TIME_OFFSET_S": 5.0,
    })
    descriptor = load_dataset(namespace["MISSION"])
    assert Path(descriptor.video) == video
    assert Path(descriptor.telemetry.path) == telemetry
    assert namespace["RESUME_BUNDLE"] is None
    assert parse_env_overrides()["ingest"]["time_offset_s"] == 5.0
