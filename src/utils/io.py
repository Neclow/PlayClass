"""
IO utilities for loading video metadata and frames.
"""

import re
import sys

from datetime import datetime
from pathlib import Path

import cv2

from loguru import logger


def setup_logger(
    log_dir: Path = Path("tmp/logs"),
    job_type: str = "sam3_hf",
    debug: bool = False,
) -> Path:
    """
    Configure loguru logger with both console and file output.

    Returns:
        Path to the log file.
    """
    level = "DEBUG" if debug else "INFO"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"{job_type}_{timestamp}.log"

    logger.remove()

    # Console handler (colored)
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
        "[<level>{level}</level>] {message}",
        level=level,
    )

    # File handler
    logger.add(
        str(log_filename),
        format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {message}",
        level=level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    return log_filename


def create_run_directory(base_output_dir: Path, job_type: str) -> Path:
    """Create a timestamped run directory for this job."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / f"{timestamp}_{job_type}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_video_run_directory(base_output_dir: Path, video_stem: str) -> Path:
    """Create a dataset-aligned run directory: ``{base}/day_{N}/{sanitized_stem}/``.

    Parses the video ID from *video_stem* via
    :func:`~src.dataset.utils.extract_video_id` to obtain the day number.
    Falls back to a flat ``{base}/{sanitized_stem}/`` if parsing fails.
    """
    from src.dataset.utils import extract_video_id

    sanitized = sanitize_filename(video_stem)
    video_id = extract_video_id(video_stem)
    if video_id is not None:
        day = video_id[5:]  # "C1G3D28" → "28"
        run_dir = base_output_dir / f"day_{day}" / sanitized
    else:
        logger.warning(
            f"Could not parse video ID from '{video_stem}'; "
            "output will not be grouped by day"
        )
        run_dir = base_output_dir / sanitized
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def sanitize_filename(name: str) -> str:
    """Sanitize a stem string for use as a directory name."""
    sanitized = re.sub(r"[^\w\-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_") or "video"


def get_video_metadata(video_path: str | Path) -> tuple[float, int]:
    """Return (fps, total_frames) for a video without loading all frames into RAM."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def load_video_frames_torchcodec(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    device: str = "cpu",
) -> list:
    """Load frames [start_frame, end_frame) as a list of RGB numpy arrays.

    Uses torchcodec for frame-accurate decoding (seek_mode="exact").
    The decoder is cached per video path so the index scan (~18 s on first
    access) only happens once across all chunks of the same video.

    When *device* is ``"cuda"``, decoding is offloaded to NVDEC hardware.
    """
    from torchcodec.decoders import VideoDecoder

    key = str(video_path)
    if key not in _torchcodec_decoder_cache:
        logger.info(
            "Building torchcodec index for '{}' (device={}).",
            Path(video_path).name,
            device,
        )
        _torchcodec_decoder_cache[key] = VideoDecoder(key, device=device)

    decoder = _torchcodec_decoder_cache[key]
    batch = decoder.get_frames_in_range(start=start_frame, stop=end_frame)
    # batch.data shape: (N, C, H, W) uint8 tensor
    return [frame.permute(1, 2, 0).cpu().numpy() for frame in batch.data]


def load_video_frames_sequential(video_path, start_frame, end_frame):
    """Load frames [start_frame, end_frame) sequentially via cv2.

    Slower than seek-based loading but frame-accurate for all codecs
    (including H.264). Use this when torchcodec is not available (e.g.
    JAX environments).
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for i in range(end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if i >= start_frame:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_video_frames_range(
    video_path: str | Path, start_frame: int, end_frame: int
) -> list:
    """Load frames [start_frame, end_frame) from a video file as a list of RGB numpy arrays.

    .. deprecated::
        Uses cv2.CAP_PROP_POS_FRAMES seeking, which is unreliable for
        H.264-encoded videos. Use :func:`load_video_frames` instead.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


_torchcodec_decoder_cache: dict[str, object] = {}

FRAME_LOADERS = {"torchcodec", "cv2_sequential", "cv2_seek"}


def load_video_frames(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    loader: str = "torchcodec",
    device: str = "cpu",
) -> list:
    """Load frames [start_frame, end_frame) using the specified loader."""
    if loader == "torchcodec":
        return load_video_frames_torchcodec(
            video_path, start_frame, end_frame, device=device
        )
    if loader == "cv2_sequential":
        return load_video_frames_sequential(video_path, start_frame, end_frame)
    if loader == "cv2_seek":
        return load_video_frames_range(video_path, start_frame, end_frame)
    raise ValueError(f"Unknown loader {loader!r}; expected one of {FRAME_LOADERS}")
