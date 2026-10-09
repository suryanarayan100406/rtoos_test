"""DRISHTI command-line interface.

The single entry point (`drishti ...`) that turns a dataset descriptor into a finished project
bundle. Commands:

    run       build a fresh bundle and execute the (subset of the) pipeline
    resume    continue an existing bundle from where it stopped (reuses its recorded config)
    inspect   print a bundle's manifest: inputs, CRS, per-stage status/timings/confidence
    verify    recompute output hashes of a bundle and check them against the manifest
    doctor    probe hardware + optional dependencies and report which stages can run here
    stages    list the canonical pipeline order

Every command fails loudly on missing inputs/dependencies (AGENTS.md §5) — nothing is faked.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .bundle import Bundle, combine
from .config import DrishtiConfig, load_config
from .config.loader import ConfigError
from .runtime import Runtime, available, detect_compute
from .stages import PIPELINE_ORDER, StageFailed, resolve_order, run_pipeline
from .stages.runner import _hash_output_path

# Keep box-drawing / status glyphs intact even when stdout is redirected to a file or pipe on a
# non-UTF-8 Windows codepage (otherwise "—", "→", "✓" become mojibake). Rich resolves sys.stdout
# lazily, so reconfiguring here — before the Console below is ever written to — still takes effect.
# Best-effort; harmless if the stream cannot be reconfigured.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="DRISHTI — single-pass drone video → accurate georeferenced 3D model (offline, ground-only).",
)
console = Console()

# --------------------------------------------------------------------------- readiness metadata
# What each stage's DEFAULT path needs to run. mode "all" => every module required; "any" => >=1.
# Pure stages (s6_global, report) have no hard requirement — they run CPU-only and merely *enhance*
# their output when an optional external tool (GLOMAP / rasterio) is present.
_STAGE_DEPS: dict[str, list[tuple[str, list[str], str]]] = {
    "s0_ingest": [("video decode", ["av", "cv2"], "any")],
    "s1_frameqa": [("ORB keyframes", ["cv2"], "all")],
    "s2_poses": [("SfM (COLMAP)", ["pycolmap"], "all"), ("CRS projection", ["pyproj"], "all")],
    "s3_masking": [("detector", ["torch", "transformers"], "all"), ("mask ops", ["cv2"], "all")],
    "s4_depth": [("depth model", ["torch", "transformers"], "all")],
    "s6_global": [("global views + CRS", ["cv2", "pyproj"], "all")],
    "s7_dense": [("TSDF fusion (Open3D)", ["open3d"], "all")],
    "s8_mesh": [("Poisson (Open3D)", ["open3d"], "all")],
    "s9_geo": [("GeoTIFF write (rasterio)", ["rasterio"], "all")],
    "s10_export": [
        ("mesh export (trimesh)", ["trimesh"], "all"),
        ("point export (laspy + pyproj)", ["laspy", "pyproj"], "all"),
        ("GeoTIFF validation (rasterio)", ["rasterio"], "all"),
    ],
    "report": [],
}

# Optional external tools that unlock a better (non-degraded) path when present.
_OPTIONAL_TOOLS = {
    "glomap": ("binary", "global SfM / bundle adjustment (s6_global)"),
    "colmap": ("binary", "SfM CLI fallback (s2_poses uses pycolmap by default)"),
    "blender": ("binary", "FBX export via headless Blender (s10_export)"),
    "PotreeConverter": ("binary", "Potree octree export (s10_export)"),
    "gtsam": ("module", "IMU/GNSS factor-graph spine tightening (s2_poses)"),
    "pdal": ("module", "SMRF ground classification for the DTM (s9_geo)"),
    "py3dtiles": ("binary", "3D Tiles export (s10_export)"),
}

_STATUS_STYLE = {
    "done": "green",
    "degraded": "yellow",
    "failed": "red",
    "running": "cyan",
    "pending": "dim",
    "skipped": "dim",
}


# --------------------------------------------------------------------------- helpers
def _probe(modules: list[str]) -> dict[str, bool]:
    return {m: available(m) for m in modules}


def _group_ok(modules: list[str], mode: str, have: dict[str, bool]) -> bool:
    present = [have[m] for m in modules]
    return all(present) if mode == "all" else any(present)


def _stage_readiness() -> tuple[dict[str, tuple[bool, str]], dict[str, bool]]:
    """Return {stage: (ready, detail)} and the flat module-availability map it was computed from."""
    all_mods = sorted({m for groups in _STAGE_DEPS.values() for _, mods, _ in groups for m in mods})
    have = _probe(all_mods)
    out: dict[str, tuple[bool, str]] = {}
    for stage in PIPELINE_ORDER:
        groups = _STAGE_DEPS.get(stage, [])
        if not groups:
            out[stage] = (True, "pure (CPU-only)")
            continue
        missing: list[str] = []
        for label, mods, mode in groups:
            if not _group_ok(mods, mode, have):
                joiner = " or " if mode == "any" else " + "
                missing.append(f"{label}: need {joiner.join(mods)}")
        out[stage] = (len(missing) == 0, "; ".join(missing) if missing else "ready")
    return out, have


def _stage_table(bundle: Bundle) -> Table:
    t = Table(box=box.SIMPLE_HEAVY, expand=False, title="Pipeline status")
    t.add_column("Stage", style="bold")
    t.add_column("Status")
    t.add_column("Env", style="dim")
    t.add_column("Wall (s)", justify="right")
    t.add_column("Conf.", justify="right")
    t.add_column("Notes")
    for name in PIPELINE_ORDER:
        rec = bundle.manifest.stage(name)
        if rec is None:
            t.add_row(name, "[dim]—[/dim]", "", "", "", "[dim]not started[/dim]")
            continue
        style = _STATUS_STYLE.get(rec.status, "white")
        conf = "" if rec.confidence.summary is None else f"{rec.confidence.summary:.2f}"
        note = rec.error or rec.degraded_reason or ""
        wall = "" if rec.wall_seconds is None else f"{rec.wall_seconds:.1f}"
        t.add_row(
            name,
            f"[{style}]{rec.status}[/{style}]",
            rec.environment or "",
            wall,
            conf,
            note if len(note) <= 70 else note[:67] + "...",
        )
    return t


def _run_header(bundle: Bundle, cfg: DrishtiConfig, rt: Runtime, dataset_name: str, order: list[str]) -> None:
    d = rt.detected
    gpu = d.cuda_name if d.cuda else "none"
    ov = "yes" if d.openvino else "no"
    compute = f"{d.cpu_count} cpu · {d.ram_gb:.0f} GB RAM · gpu: {gpu} · openvino: {ov}"
    lines = [
        f"[bold]run[/bold]      {bundle.manifest.run_id}",
        f"[bold]dataset[/bold]  {dataset_name}",
        f"[bold]profile[/bold]  {cfg.run.profile}",
        f"[bold]bundle[/bold]   {bundle.path}",
        f"[bold]compute[/bold]  {compute}",
        f"[bold]stages[/bold]   {' → '.join(order)}",
    ]
    console.print(Panel("\n".join(lines), title="DRISHTI", border_style="cyan", expand=False))

    # honest pre-flight: warn (do not block) about stages that will fail loudly here
    readiness, _ = _stage_readiness()
    blocked = [(s, readiness[s][1]) for s in order if not readiness[s][0]]
    if blocked:
        console.print(
            "[yellow]⚠ these stages cannot run in this environment and will fail loudly when reached:[/yellow]"
        )
        for s, detail in blocked:
            console.print(f"    [yellow]•[/yellow] {s} — {detail}")
        console.print(
            '[dim]  install the matching extras (e.g. pip install -e ".[recon,geo,depth]") '
            "or run those stages on the cloud-T4 tier.[/dim]"
        )


def _load_or_die(
    profile: str | None, dataset: Path | None, set_: list[str] | None, config_dir: str | None
):
    try:
        return load_config(profile=profile, dataset=dataset, set_overrides=set_, config_dir=config_dir)
    except (ConfigError, FileNotFoundError) as exc:
        console.print(f"[red]✖ config error:[/red] {exc}")
        raise typer.Exit(2) from exc
    except Exception as exc:  # pydantic ValidationError etc.
        console.print(f"[red]✖ invalid configuration:[/red] {exc}")
        raise typer.Exit(2) from exc


# --------------------------------------------------------------------------- commands
@app.command()
def run(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Dataset descriptor YAML (the input contract)."),
    profile: str | None = typer.Option(None, "--profile", "-p", help="Config profile (e.g. fast/balanced/quality)."),
    output_root: str | None = typer.Option(None, "--output-root", "-o", help="Where to create the run bundle."),
    run_id: str | None = typer.Option(None, "--run-id", help="Explicit run id (default: timestamp-based)."),
    only: str | None = typer.Option(None, "--only", help="Run only this stage (deps must be complete)."),
    upto: str | None = typer.Option(None, "--upto", help="Run from the start up to and including this stage."),
    force: bool = typer.Option(False, "--force", help="Re-run stages even if their inputs/params are unchanged."),
    set_: list[str] | None = typer.Option(None, "--set", "-s", help="Override a config key, e.g. -s dense.tsdf_voxel_m=0.03"),
    config_dir: str | None = typer.Option(None, "--config-dir", help="Directory holding default.yaml + profiles/."),
) -> None:
    """Create a fresh bundle from DATASET and run the pipeline."""
    cfg, descriptor = _load_or_die(profile, dataset, set_, config_dir)
    if descriptor is None:
        console.print("[red]✖ --dataset did not yield a descriptor[/red]")
        raise typer.Exit(2)

    try:
        order = resolve_order(only, upto)
    except KeyError as exc:
        console.print(f"[red]✖ {exc}[/red]  (known stages: {', '.join(PIPELINE_ORDER)})")
        raise typer.Exit(2) from exc

    root = output_root or cfg.run.output_root
    if run_id:
        existing = Path(root) / run_id / "manifest.json"
        if existing.is_file():
            console.print(
                f"[red]✖ a bundle already exists at {existing.parent}.[/red]  "
                f"Use [bold]drishti resume {existing.parent}[/bold] to continue it."
            )
            raise typer.Exit(2)

    bundle = Bundle.create(
        output_root=root,
        run_id=run_id,
        dataset_name=descriptor.name,
        config_snapshot=cfg.model_dump(mode="json"),
        dataset_snapshot=descriptor.model_dump(mode="json"),
        seeds={"global": cfg.run.seed},
    )
    rt = Runtime.build(cfg)
    _run_header(bundle, cfg, rt, descriptor.name, order)

    try:
        run_pipeline(bundle, cfg, rt, only=only, upto=upto, force=force)
    except StageFailed as exc:
        console.print(_stage_table(bundle))
        console.print(f"[red]✖ {exc}[/red]")
        console.print(f"[dim]partial bundle: {bundle.path} — fix the cause and `drishti resume` it.[/dim]")
        raise typer.Exit(1) from exc

    console.print(_stage_table(bundle))
    console.print(f"[green]✓ pipeline complete[/green] — bundle at [bold]{bundle.path}[/bold]")


@app.command()
def resume(
    bundle_path: Path = typer.Argument(..., help="Path to an existing run bundle."),
    only: str | None = typer.Option(None, "--only", help="Run only this stage."),
    upto: str | None = typer.Option(None, "--upto", help="Run up to and including this stage."),
    force: bool = typer.Option(False, "--force", help="Re-run stages even if fresh."),
) -> None:
    """Continue an existing bundle, reusing the config recorded in its manifest."""
    try:
        bundle = Bundle.open(bundle_path)
    except FileNotFoundError as exc:
        console.print(f"[red]✖ {exc}[/red]")
        raise typer.Exit(2) from exc

    snapshot = bundle.manifest.config_snapshot
    if not snapshot:
        console.print("[red]✖ this bundle has no config snapshot; cannot resume reproducibly.[/red]")
        raise typer.Exit(2)
    cfg = DrishtiConfig.model_validate(snapshot)
    rt = Runtime.build(cfg)

    try:
        order = resolve_order(only, upto)
    except KeyError as exc:
        console.print(f"[red]✖ {exc}[/red]")
        raise typer.Exit(2) from exc

    _run_header(bundle, cfg, rt, bundle.manifest.dataset_name or "?", order)
    try:
        run_pipeline(bundle, cfg, rt, only=only, upto=upto, force=force)
    except StageFailed as exc:
        console.print(_stage_table(bundle))
        console.print(f"[red]✖ {exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(_stage_table(bundle))
    console.print(f"[green]✓ pipeline complete[/green] — bundle at [bold]{bundle.path}[/bold]")


@app.command("inspect")
def inspect_bundle(
    bundle_path: Path = typer.Argument(..., help="Path to a run bundle."),
) -> None:
    """Print a bundle's manifest: provenance, inputs, CRS, and per-stage status."""
    try:
        bundle = Bundle.open(bundle_path)
    except FileNotFoundError as exc:
        console.print(f"[red]✖ {exc}[/red]")
        raise typer.Exit(2) from exc

    m = bundle.manifest
    head = [
        f"[bold]run[/bold]       {m.run_id}",
        f"[bold]dataset[/bold]   {m.dataset_name}",
        f"[bold]created[/bold]   {m.created_utc}",
        f"[bold]drishti[/bold]   v{m.drishti_version} (bundle v{m.bundle_version})",
    ]
    if m.crs is not None:
        head.append(f"[bold]crs[/bold]       EPSG:{m.crs.epsg} ({m.crs.derived_from}, {m.crs.vertical})")
    console.print(Panel("\n".join(head), title=f"bundle @ {bundle.path}", border_style="cyan", expand=False))

    if m.inputs:
        it = Table(box=box.SIMPLE, title="Inputs")
        it.add_column("Role", style="bold")
        it.add_column("Present")
        it.add_column("Bytes", justify="right")
        it.add_column("Path", style="dim")
        for ref in m.inputs:
            present = "[green]yes[/green]" if ref.present else "[red]no[/red]"
            size = f"{ref.bytes:,}" if ref.bytes else ""
            it.add_row(ref.role, present, size, ref.path or "—")
        console.print(it)

    console.print(_stage_table(bundle))

    done = sum(1 for s in m.stages if s.status == "done")
    degr = sum(1 for s in m.stages if s.status == "degraded")
    fail = sum(1 for s in m.stages if s.status == "failed")
    console.print(f"[dim]{done} done · {degr} degraded · {fail} failed · {len(PIPELINE_ORDER)} total stages[/dim]")


@app.command()
def verify(
    bundle_path: Path = typer.Argument(..., help="Path to a run bundle."),
) -> None:
    """Recompute output hashes and check them against what the manifest recorded (integrity check)."""
    try:
        bundle = Bundle.open(bundle_path)
    except FileNotFoundError as exc:
        console.print(f"[red]✖ {exc}[/red]")
        raise typer.Exit(2) from exc

    t = Table(box=box.SIMPLE_HEAVY, title="Integrity check")
    t.add_column("Stage", style="bold")
    t.add_column("Recorded status")
    t.add_column("Result")
    t.add_column("Detail", style="dim")

    all_ok = True
    checked = 0
    for name in PIPELINE_ORDER:
        rec = bundle.manifest.stage(name)
        if rec is None or rec.status not in ("done", "degraded"):
            continue
        checked += 1
        missing = [rel for rel in rec.outputs.values() if not bundle.artifact_path(rel).exists()]
        if missing:
            all_ok = False
            t.add_row(name, rec.status, "[red]MISSING[/red]", f"absent: {', '.join(missing)}")
            continue
        parts = [_hash_output_path(bundle.artifact_path(rel)) for rel in rec.outputs.values()]
        recomputed = combine(*parts) if parts else ""
        if recomputed == (rec.outputs_hash or ""):
            t.add_row(name, rec.status, "[green]OK[/green]", f"{len(rec.outputs)} artifact(s)")
        else:
            all_ok = False
            t.add_row(name, rec.status, "[red]MISMATCH[/red]", "outputs changed since recorded")

    console.print(t)
    if checked == 0:
        console.print("[yellow]no completed stages to verify.[/yellow]")
        raise typer.Exit(0)
    if all_ok:
        console.print("[green]✓ bundle integrity verified[/green]")
        raise typer.Exit(0)
    console.print("[red]✖ integrity check failed — see MISMATCH/MISSING rows above[/red]")
    raise typer.Exit(1)


@app.command()
def doctor() -> None:
    """Probe hardware + optional dependencies and report which stages can run in this environment."""
    d = detect_compute()
    hw = [
        f"[bold]cpu[/bold]        {d.cpu_count} logical cores",
        f"[bold]ram[/bold]        {d.ram_gb:.1f} GB",
        f"[bold]cuda[/bold]       {'yes — ' + str(d.cuda_name) + f' ({d.cuda_vram_gb:.1f} GB)' if d.cuda else 'no'}",
        f"[bold]openvino[/bold]   {'yes — ' + ', '.join(d.openvino_devices) if d.openvino else 'no'}",
        f"[bold]tier[/bold]       {'cloud-gpu session' if d.is_cloud_gpu else 'local (CPU + iGPU/NPU)'}",
    ]
    console.print(Panel("\n".join(hw), title="Detected compute", border_style="cyan", expand=False))

    readiness, have = _stage_readiness()

    dt = Table(box=box.SIMPLE, title="Optional dependency modules")
    dt.add_column("Module", style="bold")
    dt.add_column("Available")
    for mod in sorted(have):
        mark = "[green]yes[/green]" if have[mod] else "[red]no[/red]"
        dt.add_row(mod, mark)
    console.print(dt)

    st = Table(box=box.SIMPLE_HEAVY, title="Stage readiness (default path)")
    st.add_column("Stage", style="bold")
    st.add_column("Runnable here")
    st.add_column("Detail", style="dim")
    for name in PIPELINE_ORDER:
        ready, detail = readiness[name]
        mark = "[green]ready[/green]" if ready else "[red]blocked[/red]"
        st.add_row(name, mark, detail)
    console.print(st)

    ot = Table(box=box.SIMPLE, title="Optional external tools (unlock non-degraded paths)")
    ot.add_column("Tool", style="bold")
    ot.add_column("Present")
    ot.add_column("Enables", style="dim")
    for tool, (kind, why) in _OPTIONAL_TOOLS.items():
        present = (shutil.which(tool) is not None) if kind == "binary" else available(tool)
        mark = "[green]yes[/green]" if present else "[dim]no[/dim]"
        ot.add_row(tool, mark, why)
    console.print(ot)

    n_ready = sum(1 for name in PIPELINE_ORDER if readiness[name][0])
    if n_ready == len(PIPELINE_ORDER):
        console.print("[green]✓ every stage can run in this environment.[/green]")
    else:
        console.print(
            f"[yellow]{n_ready}/{len(PIPELINE_ORDER)} stages runnable here.[/yellow] "
            "Blocked stages fail loudly when reached — install their extras or use the cloud-T4 tier."
        )


@app.command()
def stages() -> None:
    """List the canonical offline pipeline order."""
    t = Table(box=box.SIMPLE, title="Pipeline order")
    t.add_column("#", justify="right", style="dim")
    t.add_column("Stage", style="bold")
    t.add_column("Requires", style="dim")
    from .stages.base import get_stage

    for i, name in enumerate(PIPELINE_ORDER):
        reqs = ", ".join(get_stage(name).requires) or "—"
        t.add_row(str(i), name, reqs)
    console.print(t)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"drishti {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    """DRISHTI — offline drone-video-to-3D pipeline."""


if __name__ == "__main__":
    app()
