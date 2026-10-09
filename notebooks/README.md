# DRISHTI notebooks

Two ways to run DRISHTI on a **free cloud T4** (Google Colab or Kaggle). Both install the *same*
`drishti` package and run the *same* CLI as every other tier — the only difference is the CUDA runtime
the host provides. Nothing is stubbed: the doctor and the run report real hardware and real results, and
**fail loudly** if a dependency, the GPU, or an input is missing. Pick by where you want the *light*
stages to run.

> **New to Kaggle? Follow [`KAGGLE-GUIDE.md`](KAGGLE-GUIDE.md)** — the complete beginner walkthrough,
> including every cell, sample/personal inputs, recovery, download, and local viewing. For the shorter
> Colab-oriented overview, see [`HOW_TO_USE.md`](HOW_TO_USE.md). This file is the notebook reference.

| Notebook | Runs on the T4 | You do locally | Use when |
|----------|----------------|----------------|----------|
| [`drishti_colab_full.ipynb`](drishti_colab_full.ipynb) | **all 11 stages** (`s0_ingest → report`) | only *view / serve* the finished bundle | you don't have the heavy deps locally and just want the whole model built for you |
| [`drishti_cloud_t4.ipynb`](drishti_cloud_t4.ipynb) | the GPU-heavy stages (`--upto s7_dense`) | `drishti resume` the remaining light stages | you have the light extras locally and want to keep the CPU stages on your machine |

---

## `drishti_colab_full.ipynb` — run everything in the hosted session

Installs the full stack on the T4 and runs the **entire pipeline end-to-end there**, then hands you the
complete bundle (model + deliverables + accuracy report + **logs**) to download. Locally you only
*view/serve* it — no heavy dependencies required.

It covers the full **import → run → export** loop: generate a realistic dataset on the T4 (or bring your
own via Drive/upload, or resume a partial local bundle), run all stages with verbose logs tee'd into the
bundle, then `inspect`/`verify` and download the whole thing (full or slim archive).

It installs **`transformers`** (S3/S4 model loading), **`laspy`** (LAS export), and xatlas UV baking
through the `depth`, `export`, and `texture` extras. Kaggle host paths take precedence over `/content`;
Option C accepts a bundle directory or full ZIP on either host, and run retries preserve console logs.
See the [Kaggle guide](KAGGLE-GUIDE.md). Optional portable **Blender 4.2** enables
the **FBX** output. It runs the **`balanced`** profile. The `max` profile selects the CUDA/gsplat + Apache
DA3 path and requires a larger CUDA runtime, so this notebook keeps `balanced` as the reproducible T4
workflow.

The §2 install cell is self-healing: it installs the CLI first and each heavy group in isolation, and —
because Open3D (needed by `s7_dense`/`s8_mesh`) has no wheel for Colab's current **Python 3.13** — it
auto-builds an isolated **Python 3.12** environment with `uv` and routes the rest of the notebook through
it, so a genuinely full run works on today's runtimes.

**Quick start**

1. Open in Colab → *Runtime → Change runtime type → GPU (T4)* (or enable the Kaggle GPU accelerator).
2. Edit the `REPO_URL` / `REPO_REF` cell (§1) to point at your DRISHTI repo.
3. Run the initial model-download permission cell and §0–§4 (GPU → code → install → doctor), then pick
   **one** import cell in §5. Use Option D for real samples or Option E for your own files.
4. Run **6B** for the aerial sample datasets (skip 6A) → §7–§9 (inspect, logs, download).
   Request GLB explicitly if desired; the default requests glTF. Do not use Run All on the unedited notebook.
5. Back home, just view it:
   ```bash
   unzip <run_id>_full.zip -d runs/
   drishti inspect runs/<run_id>          # the same manifest you saw on the T4
   drishti verify  runs/<run_id>          # confirm the download is intact
   python -m server.app                   # then browse the viewer
   ```

## `drishti_cloud_t4.ipynb` — tier split (heavy on the T4, light at home)

Runs the **GPU-heavy** stages (`s2_poses`, `s3_masking`, `s4_depth`, `s7_dense`) on the T4 via
`--upto s7_dense`, then produces a downloadable bundle you finish and serve on the CPU ground station.

**Quick start**

1. Open in Colab → *Runtime → Change runtime type → GPU (T4)* (or enable the Kaggle GPU accelerator).
2. Edit the `REPO_URL` / `REPO_REF` cell to point at your DRISHTI repo.
3. Run top to bottom; provide your `mission.yaml` (+ video/telemetry) when prompted.
4. Download the zipped bundle and **resume** the remaining stages locally:
   ```bash
   unzip <run_id>.zip -d runs/
   drishti resume runs/<run_id>           # continues at s8_mesh; the GPU stages are already 'done'
   ```

`resume` reuses the config recorded in the bundle (no `--profile`/`--dataset` flags); it skips whatever
the T4 already completed because the bundle is content-hashed and resumable.

---

Both bundles are content-hashed and byte-compatible across tiers. See
`docs/implementation/02-SYSTEM-DESIGN.md` §6 for the local-vs-cloud placement policy and
[`docs/GUIDE.md`](../docs/GUIDE.md) for the full usage guide.
