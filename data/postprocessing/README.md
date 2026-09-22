# Postprocessing

Per-video postprocessing outputs from the SAM3 best tracking run. One directory
per video, grouped by day (`day_28/`, `day_29/`), covering all 30 videos (5
cages x 3 groups x 2 days).

## Per-video files

| File                           | Source                  | Description                                                   |
| ------------------------------ | ----------------------- | ------------------------------------------------------------- |
| `bird_info.json`               | Manual                  | Bird identity assignments (track ID -> physical description). |
| `chunk_info.json`              | Tracking pipeline       | Adaptive chunking metadata (chunk boundaries, frame ranges).  |
| `tracking_postprocessing.json` | Postprocessing pipeline | Corrections applied to raw tracks (trims, merges, splits).    |
| `tracking_issues.json`         | Postprocessing pipeline | Flagged tracking problems (overlaps, identity swaps, gaps).   |
