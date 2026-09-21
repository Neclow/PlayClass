"""CLI dispatcher for the tracker evaluation (scoring) pipeline."""

import argparse

from typing import Iterable

from src.tracker_eval import (
    cvat_to_mot,
    evaluate,
    frame_selection,
    manifest,
    predictions,
)

PREPARE_STAGES = (manifest, frame_selection)
SCORE_STAGES = (cvat_to_mot, predictions, evaluate)


def _stage_defaults(stage) -> argparse.Namespace:
    """Build the default Namespace for a stage's run() — used by umbrella subcommands."""
    parser = argparse.ArgumentParser(add_help=False)
    stage._add_args(parser)
    return parser.parse_args([])


def _run_umbrella(stages: Iterable) -> None:
    for stage in stages:
        print(f"\n=== {stage.__name__.split('.')[-1]} ===")
        stage.run(_stage_defaults(stage))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pipeline.eval_tracker_all")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    manifest.add_subparser(subparsers)
    frame_selection.add_subparser(subparsers)
    cvat_to_mot.add_subparser(subparsers)
    predictions.add_subparser(subparsers)
    evaluate.add_subparser(subparsers)

    prep = subparsers.add_parser(
        "prepare",
        help="Run build-manifest + select-frames (everything before the CVAT checkpoint).",
    )
    prep.set_defaults(func=lambda _args: _run_umbrella(PREPARE_STAGES))

    score = subparsers.add_parser(
        "score",
        help="Run cvat-to-mot + convert-preds + evaluate (everything after the CVAT checkpoint).",
    )
    score.set_defaults(func=lambda _args: _run_umbrella(SCORE_STAGES))

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
