# scripts/

Helper scripts for DRISHTI. Nothing here is part of the shipped pipeline — these produce **inputs** for
it or assist development.

## `make_sample_dataset.py` — build a realistic input (video + GPS)

DRISHTI ingests a **video file + a telemetry track** (`src/drishti/stages/s0_ingest.py`) and ships **no
bundled data**. This script assembles that `MP4 + SRT` contract in one of two honest ways, writes it under
`data/<name>/` (git-ignored), and emits a matching descriptor at `configs/datasets/<name>.yaml` plus a
`PROVENANCE.txt`.

```bash
# REAL — fetch an openly-licensed aerial image set, read its real GPS EXIF,
#        and mux the real pixels + real coordinates into MP4 + DJI-style SRT.
python scripts/make_sample_dataset.py --dataset brighton_beach   # ~62 MB, 18 imgs, fast smoke test
python scripts/make_sample_dataset.py --dataset aukerman         # ~543 MB, real BUILDINGS (house+barn)
python scripts/make_sample_dataset.py --list                     # show the real registry

# SYNTHETIC — render a labelled procedural city with GROUND-TRUTH poses/GPS.
#             No network; runs anywhere; its known geometry makes accuracy checkable.
python scripts/make_sample_dataset.py --synthetic

# Then run the pipeline against the generated descriptor:
drishti run --dataset configs/datasets/brighton_beach.yaml --upto s0_ingest   # light check
drishti run --dataset configs/datasets/aukerman.yaml       --profile max      # full run (T4)
```

### What each mode is (and is not)

| Mode | Pixels | GPS | Honesty note |
|------|--------|-----|--------------|
| `--dataset <name>` | **real** (source imagery) | **real** (per-image EXIF) | Only the *playback timing* is synthesized. Imagery & coordinates are the source's own. |
| `--synthetic` | rendered | ground-truth | Labelled *synthetic* in its descriptor name **and** `PROVENANCE.txt`. Fabricates nothing about the real world — it is a known-geometry test target. |

This split is deliberate: the honesty policy (AGENTS.md §5) forbids shipping a dataset or fabricating a
GPS track, but the pipeline needs real footage to run. So the "real" mode borrows real open imagery + its
real GPS; the "synthetic" mode is clearly marked synthetic and exists to *validate* accuracy against known
truth.

### The real registry

Community OpenDroneMap example datasets (real drone imagery with GPS in EXIF). Terms vary per source repo
— **verify the source repo before redistributing** any assembled clip.

| name | approx | contents |
|------|--------|----------|
| `aukerman` | ~543 MB | Farmstead: house + barn + fields — real **buildings**. Best for a meaningful 3D model. |
| `brighton_beach` | ~62 MB | 18 images — smallest georeferenced set. Fast end-to-end smoke test. |
| `caliterra` | ~272 MB | Rolling terrain + a few structures. |
| `lewis` | ~610 MB | Larger survey, 145 images. |

### Useful flags

- `--name NAME` — override the output dataset name (default: the registry key, or `synthetic_city`).
- `--fps N` — SRT/playback frame rate (default 2 for real sets, higher for synthetic).
- `--max-width PX` — downscale real frames to this width (keeps clips small; default keeps source size).
- `--frames N`, `--width/--height`, `--seed` — synthetic-only knobs (frame count, resolution, RNG seed).

### Caching

Real downloads are cached under `data/.cache/<owner__repo>/` (git-ignored), so re-running a `--dataset`
is instant after the first fetch. Delete that folder to force a fresh download.

### Requirements

- `--synthetic`: `opencv-python-headless`, `numpy`.
- `--dataset`: additionally `pillow` (EXIF) and network access for the first fetch.

Install the pipeline's video/telemetry extras and you already have what you need:

```bash
pip install -e ".[video,telem]"
```

### Where large real sets should be fetched

`aukerman`/`lewis` are hundreds of MB and are only worth fetching where the **full** run happens — i.e. on
the cloud-T4 tier. `notebooks/drishti_cloud_t4.ipynb` has a cell that fetches a dataset and runs the heavy
stages there; on the no-dGPU dev machine, prefer `brighton_beach` or `--synthetic` for a quick check.
