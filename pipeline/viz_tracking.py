"""
Regenerate all tracking visualizations from a completed run directory.

Wraps src.viz.generate_all_visualizations, loading all inputs from the saved
artefacts in the run directory. Works for both full SAM3 tracking runs and
YOLO-scan-only runs.

Usage (from project root):
    # Single video run dir:
    pixi run -e tracker viz_tracking \
        --run-dir data/results/tracking/C1G1_Test_1_day_28_1_Camera_4_...

    # All videos under a parent dir:
    pixi run -e tracker viz_tracking --run-dir data/results/tracking

    # Point to video files (useful when the saved config path is stale):
    pixi run -e tracker viz_tracking --run-dir data/results/tracking \
        --video-dir data/videos/day_28

    # Skip annotated video rendering (plots only):
    pixi run -e tracker viz_tracking --run-dir <path> --no-video

    # Also render grounding clips (per-chunk annotated video segments):
    pixi run -e tracker viz_tracking --run-dir <path> --grounding
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import supervision as sv
import yaml
from loguru import logger

from src.tracker.scan import identify_occlusion_periods
from src.viz import generate_all_visualizations


# ---------------------------------------------------------------------------
# Video path resolution
# ---------------------------------------------------------------------------


def _resolve_video_path(
    run_dir: Path,
    override: Path | None,
    video_dir_override: Path | None = None,
) -> Path | None:
    """Resolve the video path from an explicit override, --video-dir, or the saved config.

    Resolution order:
    1. --video-path (explicit file)
    2. --video-dir + run dir name as stem
    3. video_path from saved YAML config
    4. video_dir from saved YAML config + run dir name as stem
    5. Search common locations (data/video/, /mnt/birds/rebecca2025/raw/) by stem
    """
    video_stem = run_dir.name

    if override is not None:
        video_path = override.expanduser().resolve()
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return None
        return video_path

    if video_dir_override is not None:
        candidate = video_dir_override / f"{video_stem}.mp4"
        if candidate.exists():
            return candidate
        logger.warning(f"Video not found: {candidate}")

    config_files = list(run_dir.glob("*.yaml"))
    cfg = {}
    if config_files:
        with open(config_files[0]) as f:
            cfg = yaml.safe_load(f) or {}

    raw = cfg.get("video_path")
    if raw is not None:
        video_path = Path(raw)
        if not video_path.is_absolute():
            video_path = Path.cwd() / video_path
        if video_path.exists():
            return video_path

    raw_dir = cfg.get("video_dir")
    if raw_dir is not None:
        vdir = Path(raw_dir)
        if not vdir.is_absolute():
            vdir = Path.cwd() / vdir
        candidate = vdir / f"{video_stem}.mp4"
        if candidate.exists():
            return candidate

    for search_dir in [
        Path.cwd() / "data" / "videos",
    ]:
        if not search_dir.exists():
            continue
        matches = list(search_dir.rglob(f"{video_stem}.mp4"))
        if matches:
            return matches[0]

    logger.warning(
        f"Could not find video for {video_stem}; use --video-path or --video-dir"
    )
    return None


def _get_fps(video_path: Path | None) -> float | None:
    if video_path is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else None


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_occlusion_periods(yolo_scan_df: pd.DataFrame) -> list[tuple[int, int]]:
    records = yolo_scan_df.to_dict("records")
    return identify_occlusion_periods(records)


def _is_run_dir(path: Path) -> bool:
    return (path / "chunk_info.json").exists() or (
        path / "tracking_outputs.parquet"
    ).exists()


def _find_run_dirs(root: Path) -> list[Path]:
    if _is_run_dir(root):
        return [root]
    found = []
    for p in sorted(root.rglob("chunk_info.json")):
        found.append(p.parent)
    if not found:
        for p in sorted(root.rglob("tracking_outputs.parquet")):
            found.append(p.parent)
    return found


# ---------------------------------------------------------------------------
# Annotated video rendering (streaming — decodes masks on the fly)
# ---------------------------------------------------------------------------


def _decode_rle(counts, size) -> np.ndarray:
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    return mask_util.decode({"counts": counts, "size": list(size)}).astype(np.uint8)


def _open_video_writer(
    target_path: Path, width: int, height: int, fps: float, crf: int = 23
):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(crf),
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-movflags",
                "+faststart",
                str(target_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return proc, None
    else:
        writer = cv2.VideoWriter(
            str(target_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        return None, writer


def _close_video_writer(proc, writer):
    if proc is not None:
        proc.stdin.close()
        if proc.wait() != 0:
            err = proc.stderr.read().decode("utf-8", "replace")
            raise RuntimeError(f"ffmpeg failed:\n{err}")
    if writer is not None:
        writer.release()


def _write_frame(proc, writer, frame_bgr: np.ndarray):
    if proc is not None:
        proc.stdin.write(np.ascontiguousarray(frame_bgr, dtype=np.uint8).tobytes())
    else:
        writer.write(frame_bgr)


def render_sam3_video(
    video_path: Path,
    tracking_df: pd.DataFrame,
    target_path: Path,
    fps: float,
):
    """Render SAM3 tracking masks + boxes onto the source video."""
    color_lookup = sv.ColorLookup.TRACK
    mask_annotator = sv.MaskAnnotator(color_lookup=color_lookup)
    box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=color_lookup)
    label_annotator = sv.LabelAnnotator(color_lookup=color_lookup)

    tracked_frames = set(tracking_df.index.get_level_values("frame_idx").unique())
    grouped = {fi: tracking_df.loc[fi] for fi in tracked_frames}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_path.parent.mkdir(parents=True, exist_ok=True)
    proc, writer = _open_video_writer(target_path, width, height, fps)

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in grouped:
            frame_data = grouped[frame_idx]
            n_objects = len(frame_data)

            masks = np.empty((n_objects, height, width), dtype=np.uint8)
            boxes = np.empty((n_objects, 4), dtype=np.float32)
            object_ids = np.empty(n_objects, dtype=np.int64)
            scores = np.empty(n_objects, dtype=np.float32)

            for i, (obj_id, row) in enumerate(frame_data.iterrows()):
                masks[i] = _decode_rle(row["counts"], row["size"])
                boxes[i] = row["bbox"]
                object_ids[i] = int(obj_id)
                scores[i] = float(row.get("scores", row.get("tracker_score", 0.0)))

            detections = sv.Detections(
                xyxy=boxes,
                mask=masks.astype(bool),
                confidence=scores,
                tracker_id=object_ids,
            )
            labels = [f"#{oid} {s:.2f}" for oid, s in zip(object_ids, scores)]

            frame = mask_annotator.annotate(scene=frame.copy(), detections=detections)
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(
                scene=frame, detections=detections, labels=labels
            )

        _write_frame(proc, writer, frame)

        if frame_idx % 2000 == 0 and frame_idx > 0:
            logger.info(f"    SAM3 video: {frame_idx}/{total_frames} frames")

    cap.release()
    _close_video_writer(proc, writer)
    logger.info(f"  SAM3 annotated video: {target_path} ({total_frames} frames)")


def render_yolo_video(
    video_path: Path,
    yolo_df: pd.DataFrame,
    target_path: Path,
    fps: float,
):
    """Render YOLO bbox detections onto the source video."""
    color_lookup = sv.ColorLookup.TRACK
    box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=color_lookup)
    label_annotator = sv.LabelAnnotator(color_lookup=color_lookup)

    yolo_grouped = dict(tuple(yolo_df.groupby("frame")))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_path.parent.mkdir(parents=True, exist_ok=True)
    proc, writer = _open_video_writer(target_path, width, height, fps)

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in yolo_grouped:
            frame_data = yolo_grouped[frame_idx]

            boxes = frame_data[["x1", "y1", "x2", "y2"]].values.astype(np.float32)
            confidence = frame_data["confidence"].values.astype(np.float32)
            tracker_ids = frame_data["track_id"].values.astype(np.int64)

            detections = sv.Detections(
                xyxy=boxes,
                confidence=confidence,
                tracker_id=tracker_ids,
            )
            labels = [f"#{tid} {c:.2f}" for tid, c in zip(tracker_ids, confidence)]

            frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            frame = label_annotator.annotate(
                scene=frame, detections=detections, labels=labels
            )

        _write_frame(proc, writer, frame)

        if frame_idx % 2000 == 0 and frame_idx > 0:
            logger.info(f"    YOLO video: {frame_idx}/{total_frames} frames")

    cap.release()
    _close_video_writer(proc, writer)
    logger.info(f"  YOLO annotated video: {target_path} ({total_frames} frames)")


# ---------------------------------------------------------------------------
# Grounding clip rendering (inlined from retired viz_grounding.py)
# ---------------------------------------------------------------------------

_GROUNDING_PALETTE = [
    (50, 205, 50),
    (30, 100, 255),
    (0, 165, 255),
    (200, 50, 200),
    (255, 200, 0),
    (50, 50, 220),
    (0, 215, 255),
    (180, 105, 255),
]


def _draw_grounding_frame(frame: np.ndarray, frame_rows: pd.DataFrame) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()

    for obj_id, row in frame_rows.iterrows():
        color = _GROUNDING_PALETTE[int(obj_id) % len(_GROUNDING_PALETTE)]
        mask = _decode_rle(row["counts"], row["size"])
        overlay[mask > 0] = color
        x1, y1, x2, y2 = [int(v) for v in row["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        score = row.get("scores", float("nan"))
        label = (
            f"#{obj_id}  {score:.2f}" if not np.isnan(float(score)) else f"#{obj_id}"
        )
        cv2.putText(
            out,
            label,
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
    return out


def _add_title_bar(frame: np.ndarray, text: str) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_h = 28
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[bar_h:] = frame
    cv2.putText(
        out,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def render_grounding_clips(
    run_dir: Path,
    video_path: Path,
    chunks_to_render: list[int] | None = None,
):
    """Render grounding outputs overlaid on video as per-chunk annotated clips."""
    df = pd.read_parquet(run_dir / "grounding_outputs.parquet")
    with open(run_dir / "chunk_info.json") as f:
        chunk_info_raw = json.load(f)
    chunks = chunk_info_raw["chunks"]

    if chunks_to_render is not None:
        chunks = [c for c in chunks if c["chunk_idx"] in chunks_to_render]
        if not chunks:
            logger.warning("No matching chunks found — check --chunks values.")
            return

    out_dir = run_dir / "visualizations" / "grounding_clips"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_grounding_frames = set(df.index.get_level_values("frame_idx").unique())

    for ci in chunks:
        cidx = ci["chunk_idx"]
        chunk_start, chunk_end = ci["frame_range"]

        chunk_gs_frames = sorted(
            f for f in all_grounding_frames if chunk_start <= f < chunk_end
        )
        if not chunk_gs_frames:
            continue

        gs_source = ci.get("grounding_source_frame_idx")
        logger.info(
            f"  Chunk {cidx}: grounding frames {chunk_gs_frames[0]}–{chunk_gs_frames[-1]} "
            f"({len(chunk_gs_frames)} frames)"
        )

        out_path = out_dir / f"grounding_chunk{cidx:02d}.mp4"
        proc, writer = _open_video_writer(out_path, width, height + 28, fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_gs_frames[0])
        current = chunk_gs_frames[0]

        for fi in chunk_gs_frames:
            if fi > current:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                current = fi

            ret, frame = cap.read()
            if not ret:
                current += 1
                continue
            current += 1

            if fi in df.index.get_level_values("frame_idx"):
                frame = _draw_grounding_frame(frame, df.loc[fi])

            selected_marker = " ★" if fi == gs_source else ""
            title = f"Chunk {cidx}  |  frame {fi}  |  chunk_start={chunk_start}{selected_marker}"
            frame = _add_title_bar(frame, title)
            _write_frame(proc, writer, frame)

        _close_video_writer(proc, writer)
        logger.info(f"    Saved → {out_path}")

    cap.release()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def visualize_run(
    run_dir: Path,
    video_path_override: Path | None = None,
    video_dir_override: Path | None = None,
    render_grounding: bool = False,
    render_video: bool = True,
):
    """Generate all visualizations for a single tracker run directory."""
    logger.info(f"Processing: {run_dir}")

    video_path = _resolve_video_path(run_dir, video_path_override, video_dir_override)
    fps = _get_fps(video_path)

    # --- SAM3 tracking data (optional — absent in YOLO-scan-only runs) ---
    tracking_df = None
    tracking_parquet = run_dir / "tracking_outputs.parquet"
    if tracking_parquet.exists():
        tracking_df = pd.read_parquet(tracking_parquet)
        logger.info(
            f"  tracking_outputs: {len(tracking_df)} rows, "
            f"{tracking_df.index.get_level_values('frame_idx').nunique()} frames"
        )

    # --- Per-frame metrics (optional) ---
    per_frame_df = None
    per_frame_parquet = run_dir / "metrics" / "per_frame_metrics.parquet"
    if per_frame_parquet.exists():
        per_frame_df = pd.read_parquet(per_frame_parquet)
        logger.info(f"  per_frame_metrics: {len(per_frame_df)} rows")

    # --- Chunk info (optional) ---
    chunk_info = None
    chunk_info_path = run_dir / "chunk_info.json"
    if chunk_info_path.exists():
        with open(chunk_info_path) as f:
            chunk_info = json.load(f)
        n_chunks = len(chunk_info.get("chunks", []))
        logger.info(f"  chunk_info: {n_chunks} chunks")

    # --- YOLO scan metrics (optional — for static plots) ---
    yolo_scan_df = None
    yolo_occlusion_periods = None
    yolo_parquet = run_dir / "metrics" / "yolo_scan_metrics.parquet"
    if yolo_parquet.exists():
        yolo_scan_df = pd.read_parquet(yolo_parquet)
        yolo_occlusion_periods = _load_occlusion_periods(yolo_scan_df)
        logger.info(
            f"  yolo_scan_metrics: {len(yolo_scan_df)} rows, "
            f"{len(yolo_occlusion_periods)} occlusion periods"
        )

    # --- YOLO raw detections (optional — for annotated video) ---
    yolo_tracking_df = None
    yolo_tracking_parquet = run_dir / "yolo_tracking.parquet"
    if yolo_tracking_parquet.exists():
        yolo_tracking_df = pd.read_parquet(yolo_tracking_parquet)
        logger.info(f"  yolo_tracking: {len(yolo_tracking_df)} detections")

    if tracking_df is None and yolo_scan_df is None:
        logger.warning(f"  No tracking or YOLO data found in {run_dir}, skipping")
        return

    # --- Static plots ---
    output_dir = run_dir / "visualizations"
    generate_all_visualizations(
        tracking_df=tracking_df,
        per_frame_df=per_frame_df,
        output_dir=output_dir,
        fps=fps,
        chunk_info=chunk_info,
        video_path=video_path,
        yolo_scan_df=yolo_scan_df,
        yolo_occlusion_periods=yolo_occlusion_periods,
    )
    logger.info(f"  Static plots saved to {output_dir}")

    # --- Annotated videos ---
    if render_video and video_path is not None and fps is not None:
        if tracking_df is not None:
            render_sam3_video(
                video_path,
                tracking_df,
                output_dir / "tracking_annotated.mp4",
                fps,
            )
        if yolo_tracking_df is not None:
            render_yolo_video(
                video_path,
                yolo_tracking_df,
                output_dir / "yolo_annotated.mp4",
                fps,
            )
    elif render_video and video_path is None:
        logger.warning("  Skipping annotated videos: no video path available")

    # --- Grounding clips ---
    if render_grounding:
        grounding_parquet = run_dir / "grounding_outputs.parquet"
        if grounding_parquet.exists() and video_path is not None:
            render_grounding_clips(run_dir, video_path)
        else:
            logger.warning(
                "  --grounding requested but grounding_outputs.parquet or video not found"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate tracking visualizations from a completed run directory.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Single video run dir, or parent dir to process all video subdirs",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Override video path (single run only)",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory containing .mp4 files (matched by run dir name)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip annotated video rendering (plots only)",
    )
    parser.add_argument(
        "--grounding",
        action="store_true",
        help="Also render grounding clips (per-chunk annotated video segments)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    run_dirs = _find_run_dirs(run_dir)
    if not run_dirs:
        logger.error(
            f"No tracker run directories found under {run_dir}. "
            "Expected to find chunk_info.json or tracking_outputs.parquet."
        )
        sys.exit(1)

    logger.info(f"Found {len(run_dirs)} run dir(s)")

    video_override = args.video_path
    video_dir = args.video_dir
    if video_dir is not None:
        video_dir = video_dir.expanduser().resolve()
    if video_override is not None and len(run_dirs) > 1:
        logger.warning(
            "--video-path applies to all run dirs; this only makes sense for a single run"
        )

    for rd in run_dirs:
        visualize_run(
            rd,
            video_path_override=video_override,
            video_dir_override=video_dir,
            render_grounding=args.grounding,
            render_video=not args.no_video,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
