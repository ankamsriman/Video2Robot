"""Turn a demo video into a simulated G1 repeating the task.

PYTHONPATH=. .venv/bin/python main.py --video data/raw/video_01.mp4 --step full
"""

import argparse
import logging
from pathlib import Path

from src.pipeline.run_video import (
    paths_for,
    run_keyframes,
    run_perception,
    run_planning,
    run_simulation,
)

STEPS = ("keyframes", "perception", "plan", "sim", "full")
logger = logging.getLogger("video2robot")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--step", choices=STEPS, default="full")
    args = parser.parse_args()

    paths = paths_for(args.video)
    if args.step in ("keyframes", "full"):
        times = run_keyframes(args.video, paths)
        logger.info("Key moments at: %s", [round(time, 2) for time in times])
    if args.step in ("perception", "full"):
        run_perception(args.video, paths)
        logger.info("Saved item tracks to %s", paths.tracks)
    if args.step in ("plan", "full"):
        demo = run_planning(paths)
        for number, step in enumerate(demo.plan.steps, start=1):
            logger.info(
                "Step %d: %s %s %s",
                number,
                step.object,
                step.relation.value,
                step.reference,
            )
    if args.step in ("sim", "full"):
        result = run_simulation(paths)
        logger.info("Success: %s, attempts: %s", result["success"], result["attempts"])
        logger.info("Saved video to %s", paths.rollout)


if __name__ == "__main__":
    main()
