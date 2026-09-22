# Tracker evaluation

Benchmark of the 7-way tracker ablation over 5 videos with sparse CVAT-annotated
keyframes, scored with motmetrics + TrackEval. Compares SAM 3 variants against
Grounded-SAM-2 and YOLO baselines.

```sh
# Prepare video manifest + keyframe schedule
pixi run -e tracker prepare_tracker

# Convert tracker outputs to MOT format, then score against CVAT ground truth
pixi run -e tracker score_tracker
```

## Modules

| Module               | Subcommand         | Purpose                                                                               |
| -------------------- | ------------------ | ------------------------------------------------------------------------------------- |
| `manifest.py`        | `build-manifest`   | Rank candidate videos by YOLO-scan difficulty; mark the 5 eval picks.                 |
| `frame_selection.py` | `select-frames`    | Emit the per-video keyframe schedule (chunk-guided + occlusion-bracketing + uniform). |
| `cvat_to_mot.py`     | `cvat-to-mot`      | Convert the CVAT project backup to one sparse MOTChallenge 1.1 file per video.        |
| `predictions.py`     | `convert-preds`    | Convert each variant's tracker output parquet to MOTChallenge `.txt`.                 |
| `evaluate.py`        | `evaluate`         | Score all variants with TrackEval + py-motmetrics.                                    |
| `__main__.py`        | `prepare`, `score` | Umbrella entry points.                                                                |

## Evaluation subset

Five 15-min videos from days 28 and 29, one per cage, with the hardest group per
cage selected from the YOLO scan via a composite difficulty score. Recorded in
`data/results/eval_tracking/video_manifest.csv` (`selected=True` rows). The 5
video stems are also listed in `config/benchmark_videos.txt` for `--eval` mode.

## Sparse ground truth

Ground truth is sparse by design: 87-88 human-verified frames per video (462
total). At each keyframe every visible bird has a human-drawn bbox. Standard MOT
metrics (HOTA, IDF1, MOTA) are defined over GT-present frames and impose no
density assumption.

The keyframe schedule combines three sources, deduplicated in priority order:

1. **Chunk-guided** — for every internal adaptive-chunk boundary `B`, sample
   `{B-5, B, B+5}`.
2. **Occlusion-bracketing** — for the three longest occlusion periods `(a, b)`,
   sample `{a-3, a, floor((a+b)/2), b, b+3}`.
3. **Uniform** — one frame every 30 s as a temporal backbone.

Schedules: `data/results/eval_tracking/annotation_frames.csv`.

## Variants scored

| Label                  | Variant  | Config                            |
| ---------------------- | -------- | --------------------------------- |
| `A_yolo_botsort`       | A        | `sam3_best` (YOLO scan byproduct) |
| `A1_yolo_botsort_reid` | A1       | `yolo_botsort_reid_on`            |
| `B_gs2_strict`         | B-strict | `gs2_baseline`                    |
| `B_gs2_fixed`          | B-parity | `gs2_adaptive_recovery`           |
| `C_sam3_frame_zero`    | C-strict | `sam3_baseline`                   |
| `D_sam3_fixed`         | D        | `sam3_adaptive_grounding`         |
| `E_sam3_adaptive`      | E        | `sam3_best`                       |

All variants use raw automated tracker outputs (no manual postprocessing).

## CVAT annotation workflow

The pipeline brackets an offline CVAT annotation step: everything before is the
**prepare** stage (manifest + keyframe schedule), everything after is the
**score** stage (MOT export + prediction conversion + evaluation).

### Handoff to annotator

| File                                               | Purpose                                                               |
| -------------------------------------------------- | --------------------------------------------------------------------- |
| `data/results/eval_tracking/annotation_frames.csv` | The 462 keyframes to annotate.                                        |
| `data/results/eval_tracking/video_manifest.csv`    | Selection metadata including source MP4 paths.                        |
| The 5 `.mp4` files                                 | Listed in `video_manifest.csv` under `path` for `selected=True` rows. |

### CVAT setup

Create a project `playclass-tracker-eval` with a single label `bird`. One Task
per video (5 total), task name = `video_id` from the manifest. Upload the MP4
directly (do not convert to image set).

**Workflow:** switch to Track mode. For each bird, draw a tight bbox at the
first keyframe (creates a Track), then visit every subsequent listed keyframe
and adjust. Only visit frames in `annotation_frames.csv`.

**Conventions:** partial occlusion: annotate the full estimated bbox. Total
occlusion / bird out of frame: mark `outside=true` (shortcut **O**); resume on
the same Track when the bird reappears.

### Handoff back

Export a CVAT project Backup (Project → Actions → Backup project). Extract under
`data/results/eval_tracking/tracker_benchmark/cvat_backup/playclass-tracker-eval/`.
The converter reads the native `annotations.json` and keeps only human-drawn,
non-`outside` rectangles (no interpolated frames).

## Scoring

```sh
pixi run -e tracker score_tracker
```

Runs `cvat-to-mot` → `convert-preds` → `evaluate` in sequence. Outputs:

| File                            | Granularity                              |
| ------------------------------- | ---------------------------------------- |
| `results/metrics_per_video.csv` | One row per (variant, video).            |
| `results/metrics_per_cage.csv`  | Aggregated per (variant, cage).          |
| `results/metrics_aggregate.csv` | One row per variant (whole eval subset). |

All under `data/results/eval_tracking/`.

Metrics: **HOTA family** (HOTA, DetA, AssA, LocA) and **py-motmetrics** (IDF1,
MOTA, MOTP, ID switches). Matching uses IoU 0.5. Missing predictions are scored
as zero-prediction penalty.
