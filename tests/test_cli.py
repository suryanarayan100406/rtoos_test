"""CLI smoke tests — wiring, exit codes, and honest failure paths (no heavy deps needed)."""
from typer.testing import CliRunner

from drishti.cli import app

runner = CliRunner()


def test_version():
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0
    assert "drishti" in res.output


def test_stages_lists_pipeline():
    res = runner.invoke(app, ["stages"])
    assert res.exit_code == 0
    assert "s0_ingest" in res.output
    assert "report" in res.output


def test_doctor_runs_and_lists_stages():
    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 0
    assert "s0_ingest" in res.output          # stage-readiness table


def test_inspect_missing_bundle_exit_2():
    res = runner.invoke(app, ["inspect", "no_such_bundle_dir"])
    assert res.exit_code == 2


def test_verify_missing_bundle_exit_2():
    res = runner.invoke(app, ["verify", "no_such_bundle_dir"])
    assert res.exit_code == 2


def test_run_missing_dataset_exit_2():
    res = runner.invoke(app, ["run", "--dataset", "no_such_dataset.yaml", "--config-dir", "configs"])
    assert res.exit_code == 2


def test_run_bad_stage_name_exit_2(tmp_path):
    # a valid dataset file, but an unknown --only stage should fail before doing work
    ds = tmp_path / "ds.yaml"
    ds.write_text(
        "name: t\nvideo: v.MP4\ntelemetry:\n  format: dji_srt\n  path: v.SRT\n", encoding="utf-8"
    )
    res = runner.invoke(app, ["run", "--dataset", str(ds), "--only", "not_a_stage", "--config-dir", "configs"])
    assert res.exit_code == 2
