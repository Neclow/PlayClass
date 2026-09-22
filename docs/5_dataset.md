# Dataset Build

| Script | Description |
|--------|-------------|
| `script/build_dataset.py` | Postprocess tracking outputs, match bird IDs, build dataset parquets |
| `script/extract_features.py` | Extract mask features + window summaries from dataset tracks (CPU) |

---

## Overview

The dataset is built in three steps from two sources: the SAM3 tracking outputs (`tracking_outputs.parquet` per video) and the registration protocol Excel files (behaviour labels + bird identity).

**Step 1** (`build_dataset`) is lightweight and produces the canonical `tracks.parquet` and `labels.parquet`. Steps 2 and 3 are slow and independent of each other — run them in parallel if you have the resources.

---

## Step 1 — Labels, postprocessing, windows

```sh
pixi run -e dataset build_dataset
```

**What it does:**

1. **Parses labels** from the registration Excel files — behaviour annotations (locomotor play, object play, no-play) timestamped per bird per video.
2. **Applies postprocessing** — reads each video's `tracking_postprocessing.json`, trims bad frame ranges, merges split tracks (`id_switch`), and renames tracker IDs to protocol bird IDs (`id_match`). See [postprocessing](4_postprocessing.md) for how to fill these in.
3. **Assigns temporal windows** — each behaviour label defines a fixed-duration window; track frames are assigned to the window they fall in.
4. **Filters incomplete windows** — windows where too few frames remain after trimming (default: <50% of expected frames) are dropped from both tracks and labels to avoid degenerate feature statistics.

**Output:** `data/dataset/tracks.parquet` + `data/dataset/labels.parquet`, keyed by `(video_id, bird_id, window)`.

---

## Step 2 — Mask features (CPU)

```sh
pixi run -e dataset extract_features
```

**What it does:** Decodes the RLE masks in `tracks.parquet` frame-by-frame and computes six handcrafted features per bird per frame:

| Feature | What it captures |
|---------|-----------------|
| `mask_area` | Body size / distance from camera |
| `aspect_ratio` | Pose (upright vs. horizontal) |
| `velocity` | Movement speed |
| `area_change_rate` | Rapid size changes (occlusion, pose shifts) |
| `min_dist_to_other` | Proximity to nearest pen-mate |
| `mean_dist_to_other` | Average spread across pen |

These are then summarized per window (mean, std, min, max, median → `features_windowed.parquet`) and also stored as frame-binned temporal tensors (`features_binned.pt`) for temporal models.

---

## Step 3 — Embeddings (GPU)

```sh
# DINOv3 ViT-L (default)
pixi run -e embeddings extract_embeddings_dinov3

# V-JEPA 2.1 ViT-L temporal
pixi run -e embeddings python -m script.extract_embeddings_vjepa2 --temporal

# VideoPrism Base temporal
pixi run -e videoprism extract_videoprism --temporal
```

**What it does:** For each `(video_id, bird_id, window)`, crops the bird's bounding box from video frames and runs them through a pretrained visual backbone. The result is a per-window embedding tensor of shape `(frames_in_window, D)`.

These capture appearance and motion information the handcrafted features miss — coat colour, posture details, fine-grained motion texture.

Outputs are saved as `.pt` dicts keyed by `(video_id, bird_id, window)`, e.g. `embeddings_dinov3_vitl.pt`.

---

## Output files

| File | Keyed by | Contents |
|------|----------|----------|
| `tracks.parquet` | `(video_id, bird_id, frame_idx)` | Cleaned tracks with RLE masks, bbox, window assignment |
| `labels.parquet` | `(video_id, bird_id, window)` | Behaviour labels aligned to track coverage |
| `features_all.parquet` | `(video_id, bird_id, frame_idx)` | Per-frame mask features |
| `features_windowed.parquet` | `(video_id, bird_id, window)` | Per-window feature summaries |
| `features_binned.pt` | `(video_id, bird_id, window)` | Temporal feature tensors (same format as embeddings) |
| `embeddings_*.pt` | `(video_id, bird_id, window)` | Visual backbone embeddings, shape `(F_w, D)` |
