# Tracker evaluation benchmark

Evaluation of the 7-way tracker ablation against human-annotated ground truth.
The pipeline has two phases separated by a manual CVAT annotation step.

## Pipeline

1. **Prepare** (`pixi run -e tracker prepare_tracker`)
   - `build-manifest` — ranks 30 candidate videos by difficulty, selects 5.
   - `select-frames` — picks annotation keyframes (chunk-guided + occlusion +
     uniform).
2. **Annotate** — manual bounding-box annotation in CVAT on the selected
   keyframes.
3. **Score** (`pixi run -e tracker score_tracker`)
   - `cvat-to-mot` — converts CVAT backup to sparse MOTChallenge ground truth.
   - `convert-preds` — converts tracker parquets to MOTChallenge `.txt` files.
   - `evaluate` — computes HOTA, MOTA, IDF1, etc. per video/cage/aggregate.

## Directory layout

| Path                                 | Description                                                                     |
| ------------------------------------ | ------------------------------------------------------------------------------- |
| `video_manifest.csv`                 | All 30 videos ranked by difficulty; `selected=True` for the 5 benchmark videos. |
| `annotation_frames.csv`              | Selected keyframes per benchmark video (frame index + source).                  |
| `annotation_frames_summary.csv`      | Per-video summary of frame selection (counts by source).                        |
| `tracker_benchmark/cvat_backup/`     | Raw CVAT project backup (human annotations).                                    |
| `tracker_benchmark/ground_truth/`    | Sparse MOTChallenge ground truth (one `.txt` per video).                        |
| `tracker_benchmark/predictions_mot/` | Converted predictions per variant per video.                                    |
| `results/`                           | Final metrics CSVs (per-video, per-cage, aggregate).                            |

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
