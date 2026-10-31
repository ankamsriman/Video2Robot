"""Print `colors:` settings for configs/default.yaml, from points on a video frame.

PYTHONPATH=. .venv/bin/python scripts/read_colors.py --video data/raw/video_01.mp4 \
    --point apple=640,400 --point lemon=700,410 --save-frame first_frame.jpg
"""

import argparse
import logging
from pathlib import Path

import cv2

from src.perception.read_colors import suggest_spec

PATCH_RADIUS = 8
logger = logging.getLogger("read_colors")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--point", action="append", default=[], help="name=x,y")
    parser.add_argument("--save-frame", type=Path)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {args.video}")
    if args.save_frame:
        cv2.imwrite(str(args.save_frame), frame)

    logger.info("colors:")
    for point in args.point:
        name, position = point.split("=")
        x, y = (int(value) for value in position.split(","))
        sample = frame[
            y - PATCH_RADIUS : y + PATCH_RADIUS, x - PATCH_RADIUS : x + PATCH_RADIUS
        ]
        suggested = suggest_spec(name, sample)
        spec = suggested.spec
        logger.info(
            "  %s: { hue: [%d, %d], min_saturation: %d, min_value: %d }%s",
            name,
            spec.hue[0],
            spec.hue[1],
            spec.min_saturation,
            spec.min_value,
            "" if suggested.is_colorful else "  # too pale to track reliably",
        )


if __name__ == "__main__":
    main()
