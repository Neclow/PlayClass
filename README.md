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
| 2. Tracking | [docs/2_tracking.md](docs/2_tracking.md) | `tracker`, `gs2` |
| 3. Tracker evaluation *(optional)* | [docs/3_tracker_eval.md](docs/3_tracker_eval.md) | `tracker`, `gs2` |
| 4. Postprocessing | [docs/4_postprocessing.md](docs/4_postprocessing.md) | `tracker` |
| 5. Build dataset | [docs/5_dataset.md](docs/5_dataset.md) | `tracker` |
| 6. Embeddings | [docs/6_embeddings.md](docs/6_embeddings.md) | `embeddings`, `videoprism` |
| 7. Classification | [docs/7_classification.md](docs/7_classification.md) | `classifier` |

### Analysis

| Stage | Docs | Environment |
|-------|------|-------------|
| 8. Analysis (clustering, classification figures, feature attribution) | [docs/8_analysis.md](docs/8_analysis.md) | `classifier` |

### Tests

```sh
# 1. Labels, postprocessing, windows (fast, ~seconds)
pixi run -e dataset build_dataset

pixi run test_features                    # Feature extraction unit tests (pytest)
pixi run test_postprocessing              # Postprocessing logic unit tests (pytest)
pixi run test_labels                      # Label parsing unit tests (pytest)
pixi run test_post_build                  # Data integrity checks on a built dataset (pytest; skipped if no dataset)
```
