# PlayClass

A pipeline for play behaviour recognition in videos of poultry with tracking, postprocessing, feature extraction and classification.

## Installation

### Prerequisites

All code was developed and tested on Ubuntu 24.04 (linux-64) with CUDA 12.6.

### Dependencies

Run the following commands to setup the main dependencies for this project.

```sh
git submodule update --init --recursive
pixi install
```

- To install SAM 3, you will need apply for approval at: <https://huggingface.co/facebook/sam3>
- Pixi environments:
  - `default` (base)
  - `tracker` (SAM3, tracker evaluation)
  - `gs2` (Grounded-SAM-2; tracker evaluation)
  - `dataset` (build dataset + features)
  - `embeddings` (DINOv3/V-JEPA)
  - `videoprism` (JAX)
  - `classifier` (training, evaluation)

Pixi environments: `default` (base), `tracker` (SAM3), `dataset` (build + features), `embeddings` (DINOv3/V-JEPA), `classifier` (training), `videoprism` (JAX), `gs2` (Grounded-SAM-2; used for tracker benchmarking), `tracker-evaluation` (CPU-only tracker scoring; motmetrics + pycocotools). Platform is Linux-only (CUDA 12.6).

## Data

```
data/
  labels/          Registration protocol Excel files (behaviour labels + bird info)
  tracking/        Symlinks to tracking run output dirs (gitignored)
  postprocessing/  Version-controlled per-video postprocessing JSONs + parquets (day_28/, day_29/)
  tracker_eval/    Version-controlled tracker benchmark artefacts (video manifest, keyframes, ablation configs, scored results)
ext-data/          Symlink to large data outputs (results, image sequences, embeddings, etc.)
```

| Stage | Docs | Environment |
| ------- | ------ | ------------- |
| 1. Data (TO-DO: Zenodo deposit in preparation) | [data/README.md](data/README.md) | — |
| 2. Tracker | [docs/2_tracker.md](docs/2_tracker.md) | `tracker` |
| 2a. Tracker evaluation *(optional)* | [docs/2a_tracker_eval.md](docs/2a_tracker_eval.md) | `tracker`, `gs2` |
| 3. Postprocessing | [docs/3_postprocessing.md](docs/3_postprocessing.md) | `dataset` |
| 4. Build dataset | [docs/4_dataset.md](docs/4_dataset.md) | `dataset` |
| 5. Embeddings | [docs/5_embeddings.md](docs/5_embeddings.md) | `embeddings`, `videoprism` |
| 6. Classification | [docs/6_classification.md](docs/6_classification.md) | `classifier` |

Built from tracking outputs + registration protocol Excel files in three steps:

```sh
# 1. Labels, postprocessing, windows (fast, ~seconds)
pixi run -e dataset build_dataset

pixi run -e dataset test_features         # Feature extraction unit tests (pytest)
pixi run -e dataset test_postprocessing   # Postprocessing logic unit tests (pytest)
pixi run -e dataset test_post_build       # Data integrity checks on a built dataset (pytest; skipped if no dataset)
```
