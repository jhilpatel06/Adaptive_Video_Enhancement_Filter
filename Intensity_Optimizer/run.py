from __future__ import annotations

import argparse

from src.adaptive_enhancer.config import EnhancementConfig
from src.adaptive_enhancer.pipeline import VideoEnhancementPipeline


def parse_source(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run adaptive real-time video enhancement on a webcam or video file."
    )
    parser.add_argument("--source", default="0", help="Webcam index or path to video file.")
    parser.add_argument("--target-fps", type=int, default=30, help="Target FPS for adaptive performance decisions.")
    parser.add_argument("--max-width", type=int, default=760, help="Resize each input frame wider than this value.")
    parser.add_argument("--histogram-height", type=int, default=125, help="Height of the histogram panel.")
    parser.add_argument("--no-overlay", action="store_true", help="Hide live metrics overlay.")
    parser.add_argument(
        "--enhanced-only",
        action="store_true",
        help="Show only the enhanced video instead of the side-by-side comparison.",
    )
    parser.add_argument("--no-histogram", action="store_true", help="Hide the live intensity histogram panel.")
    parser.add_argument("--no-heatmap", action="store_true", help="Hide the live original/enhanced intensity heatmap panel.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = EnhancementConfig(
        target_fps=args.target_fps,
        max_width=args.max_width,
        histogram_height=args.histogram_height,
        show_overlay=not args.no_overlay,
        side_by_side=not args.enhanced_only,
        show_histogram=not args.no_histogram,
        show_heatmap=not args.no_heatmap,
    )
    pipeline = VideoEnhancementPipeline(config)
    pipeline.run(parse_source(args.source))


if __name__ == "__main__":
    main()
