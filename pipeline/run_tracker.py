"""Unified launcher for tracking pipelines (SAM 3, Grounded-SAM-2)."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = (
    "expandable_segments:True,garbage_collection_threshold:0.6"
)

from argparse import ArgumentParser
from pathlib import Path

from omegaconf import OmegaConf

from src._config import DEFAULT_TRACKING_DIR, DEFAULT_VIDEO_DIR
from src.io import create_video_run_directory

DEFAULT_CONFIG = "config/sam3_best.yaml"


def _infer_backend(cfg) -> str:
    if "gs2" in cfg:
        return "gs2"
    return "sam3"


def _get_backend_module(backend: str):
    if backend == "sam3":
        from src.tracking.sam3 import _run_batch, _run_single_video
    elif backend == "gs2":
        from src.tracking.grounded_sam_2 import _run_batch, _run_single_video
    else:
        raise ValueError(f"Unknown backend: {backend!r}")
    return _run_batch, _run_single_video


def run(cfg, config_path: str | Path, backend: str, video_path: str | Path) -> None:
    config_path = Path(config_path)
    video_path = Path(video_path)

    _run_batch, _run_single_video = _get_backend_module(backend)

    batch_dir = Path(DEFAULT_TRACKING_DIR) / config_path.stem
    batch_dir.mkdir(parents=True, exist_ok=True)

    if video_path.is_file():
        run_dir = create_video_run_directory(batch_dir, video_path.stem)
        _run_single_video(cfg, run_dir, config_path=config_path)
    elif video_path.is_dir():
        _run_batch(cfg, batch_dir, video_path, config_path=config_path)
    else:
        raise FileNotFoundError(f"Video path does not exist: {video_path}")


def main():
    parser = ArgumentParser(description="Tracking pipeline launcher")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to tracking config file.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help="Video file or directory of videos.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    backend = _infer_backend(cfg)

    run(cfg, config_path=args.config, backend=backend, video_path=args.video_path)


if __name__ == "__main__":
    main()
