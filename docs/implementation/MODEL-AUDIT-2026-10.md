# Model ecosystem audit — 2026-10

This is a research and licensing audit, not a benchmark claim. DRISHTI only promotes a model into the
default commercial path after its checkpoint license, reproducible reference, and aerial measurements are
recorded.

## Candidates reviewed

| Candidate | Upstream evidence | Commercial checkpoint | Relevant capability | DRISHTI decision |
|---|---|---|---|---|
| MapAnything | `facebookresearch/map-anything` README and Apache repository | `facebook/map-anything-apache`, Apache-2.0 | metric pointmaps, depth, confidence, calibration/pose-conditioned multi-view inference, memory-efficient windows | registered as optional `mapanything_apache`; not balanced default until aerial validation |
| Depth Anything 3 | `ByteDance-Seed/Depth-Anything-3` README/model cards | DA3-SMALL, DA3-BASE, DA3METRIC-LARGE, Apache-2.0 | multi-view depth/pose, pose-conditioned depth, metric monocular depth, confidence, DA3 streaming | registered and wired through a real optional adapter; `max` profile uses DA3-BASE |
| Depth Anything V2 | official repository + Transformers model documentation | Small HF checkpoint Apache-2.0; Base/Large upstream checkpoints CC-BY-NC | robust relative depth, light T4/local path | Small is the balanced baseline because DRISHTI's S4 scale-aligns it to the metric spine; Base is reference-only |
| gsplat | official `nerfstudio-project/gsplat` repository/docs | Apache-2.0 | CUDA differentiable Gaussian rasterization, RGB/depth render modes, packed memory path, multi-GPU support upstream | registered and wired for the optional genuine Gaussian dense route |

## Constraints that affect selection

- **DA3 license is variant-specific.** The upstream model table marks the DA3 base/small and metric-large
  checkpoints Apache-2.0 while the larger giant/large/nested checkpoints are CC-BY-NC. DRISHTI therefore
  does not silently load the larger research checkpoints.
- **MapAnything has two checkpoint families.** The default upstream checkpoint is CC-BY-NC; the Apache
  checkpoint is the commercial candidate and is the only one registered for shipped use.
- **gsplat is CUDA-first.** Its current upstream main branch documents newer PyTorch/CUDA requirements
  and capabilities than the free T4 baseline. DRISHTI keeps it cloud-only and does not claim local support.
- **Model cards are not aerial validation.** None of the upstream pages is sufficient evidence for the
  <=1 m DRISHTI target on single-pass drone footage. That number remains a measured acceptance gate.

## Reproducibility fields now tracked

The model registry now records checkpoint, checkpoint license, commercial status, source, runtime/device
class, approximate resource envelope, precision, download source, and offline compatibility. The registry
pins the upstream Hugging Face revision and model-weight LFS SHA-256 for each promoted checkpoint; the
weight files themselves are not committed. A deployed run must still record the resolved cache path and
verify that checksum before inference.
