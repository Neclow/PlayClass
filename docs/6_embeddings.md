# Embeddings Extraction

| Script                                      | Env          | Description                                  |
| ------------------------------------------- | ------------ | -------------------------------------------- |
| `pipeline/extract_embeddings_dinov3.py`     | `embeddings` | DINOv3 CLS-token embeddings (image backbone) |
| `pipeline/extract_embeddings_vjepa2.py`     | `embeddings` | V-JEPA 2 / 2.1 video embeddings              |
| `pipeline/extract_embeddings_videoprism.py` | `videoprism` | VideoPrism video embeddings (JAX)            |

All three scripts read `tracks.parquet` from the dataset dir, load video frames,
and save a `.pt` dict keyed by `(video_id, bird_id, window)`.

---

## DINOv3 (image backbone)

No setup needed — model weights download automatically from HuggingFace on first
run.

```sh
# Default: ViT-L, bbox crop
pixi run -e embeddings extract_embeddings_dinov3 \
    --video-dir data/videos/day_28 data/videos/day_29

# ViT-B backbone
pixi run -e embeddings extract_embeddings_dinov3 \
    --video-dir data/videos/day_28 data/videos/day_29 \
    --model-name facebook/dinov3-vitb16-pretrain-lvd1689m

# Custom resolution (DINOv3 was trained at 256; supports up to 768)
pixi run -e embeddings extract_embeddings_dinov3 \
    --video-dir data/videos/day_28 data/videos/day_29 --resolution 256
```

Output: `embeddings_dinov3_vitl.pt` (or `vitb`, `r256`, etc. — name
auto-generated from args).

---

## V-JEPA 2 / 2.1 (video backbone)

### V-JEPA 2 (HuggingFace) — no setup needed

```sh
pixi run -e embeddings python -m pipeline.extract_embeddings_vjepa2 \
    --video-dir data/videos --device cuda:0 --temporal
```

### V-JEPA 2.1 (torch.hub) — one-time setup required

V-JEPA 2.1 checkpoints are not on HuggingFace; they're downloaded separately and
loaded via `torch.hub`. The setup script also patches a namespace collision
between the hub repo's `src/` directory and this project's own `src/` package.

```sh
# Download checkpoint + patch hub cache (run once)
bash scripts/setup_vjepa2.1.sh                   # default: vjepa2_1_vit_large_384
bash scripts/setup_vjepa2.1.sh vjepa2_1_vit_base_384   # ViT-B variant
```

Available models:

| Model name                  | Size                         |
| --------------------------- | ---------------------------- |
| `vjepa2_1_vit_base_384`     | ViT-B (distilled from ViT-G) |
| `vjepa2_1_vit_large_384`    | ViT-L (default)              |
| `vjepa2_1_vit_giant_384`    | ViT-G                        |
| `vjepa2_1_vit_gigantic_384` | ViT-G2                       |

Then extract:

```sh
pixi run -e embeddings python -m pipeline.extract_embeddings_vjepa2 \
    --video-dir data/videos/day_28 data/videos/day_29 \
    --device cuda:0 --temporal \
    --model-name vjepa2_1_vit_large_384
```

Output: `embeddings_vjepa21_vitl_temporal.pt` (name auto-generated from args).

---

## VideoPrism (JAX)

Runs in its own `videoprism` pixi env (JAX + TensorFlow, separate from the
PyTorch `embeddings` env). TF is imported only to be blocked from grabbing the
GPU — the actual inference runs in JAX.

No extra setup: model weights download automatically on first run via the
`videoprism` package.

```sh
# Temporal embeddings (default: ViT-B)
pixi run -e videoprism extract_videoprism \
    --video-dir data/videos/day_28 data/videos/day_29 --device 0 --temporal

# Raw patch tokens for a trainable pooler (~88 GB output — large!)
pixi run -e videoprism extract_videoprism \
    --video-dir data/videos/day_28 data/videos/day_29 --device 0 --raw
```

Output: `embeddings_videoprism_temporal.pt` or `embeddings_videoprism_raw.pt`.

---

## Crop modes

All three scripts support `--crop-mode` to control what region of the frame is
extracted:

| Mode                    | Description                                                     |
| ----------------------- | --------------------------------------------------------------- |
| `bbox`                  | Per-frame bounding box (default)                                |
| `plain256` / `plain384` | Fixed square around the bbox centroid                           |
| `union512` / `union384` | Fixed square around the centroid of all three birds' union bbox |
| `maskbbox`              | Bbox crop with non-bird pixels blacked out                      |
| `dimbbox`               | Same, but dimmed to 40%                                         |
| `silbbox`               | Flat silhouette (shape only, no texture)                        |

Mask-based modes (`maskbbox`, `dimbbox`, `silbbox`) require the RLE mask columns
in `tracks.parquet` and are matched controls against the default `bbox` crop.

The crop mode is appended to the output filename: e.g.
`embeddings_vjepa21_vitl_maskbbox_temporal.pt`.
