"""Watch the robot repeat one of my videos live in the MuJoCo window, to see how it moves.

The window opens when the robot starts. Drag to turn the camera, scroll to zoom, and close
the window when done. Claude's saved points are reused, so no API call is made.

On a Mac, run it with mjpython:
PYTHONPATH=. .venv/bin/mjpython scripts/watch_video_live.py --video data/raw/task2_triangle_check.mov --claude --choose-arms --slow 2
"""

import argparse
import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mujoco
import mujoco.viewer
import yaml

from scripts import glass_from_video
from src.perception.glass_tracker import GlassVideoConfig
from src.simulation.recorder import CAMERA
from src.simulation.scene import ROOT

POINTS = ROOT / "data/processed/task2_setups/claude_points.json"
IDLE_SLEEP = 0.05
logger = logging.getLogger(__name__)


def live_window(slow: float):
    """Stand in for the video recorder: show each frame in a window, at a watchable speed."""

    @contextmanager
    def window(model: mujoco.MjModel, path: Path) -> Iterator:
        state: dict = {}

        def on_frame(data: mujoco.MjData) -> None:
            if "viewer" not in state:
                viewer = mujoco.viewer.launch_passive(model, data)
                for key, value in CAMERA.items():
                    if key == "lookat":
                        viewer.cam.lookat[:] = value
                    else:
                        setattr(viewer.cam, key, value)
                state.update(viewer=viewer, start=time.monotonic() - data.time * slow)
            viewer = state["viewer"]
            if not viewer.is_running():
                return
            viewer.sync()
            ahead = data.time * slow - (time.monotonic() - state["start"])
            if ahead > 0:
                time.sleep(ahead)

        yield on_frame
        viewer = state.get("viewer")
        logger.info("Done. Close the MuJoCo window to finish.")
        while viewer is not None and viewer.is_running():
            time.sleep(IDLE_SLEEP)

    return window


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--claude", action="store_true", help="use Claude's saved points")
    parser.add_argument("--two-arms", action="store_true")
    parser.add_argument("--choose-arms", action="store_true")
    parser.add_argument("--slow", type=float, default=1.0, help="2 plays at half speed")
    args = parser.parse_args()
    config = GlassVideoConfig(
        **yaml.safe_load(glass_from_video.CONFIG_PATH.read_text())["real_video"]
    )
    saved = json.loads(POINTS.read_text()) if args.claude else {}
    start = [tuple(p) for p in saved.get(args.video.stem, [])] or None
    end = [tuple(p) for p in saved.get(f"{args.video.stem}@end", [])] or None
    demo, is_flip = glass_from_video.read_video(args.video, config, start, end)
    glass_from_video.video_recorder = live_window(args.slow)
    report = glass_from_video.run_robot(
        args.video, demo, is_flip, False, False, config,
        two_arms=args.two_arms, suffix_extra="_live", choose_arms=args.choose_arms,
    )  # fmt: skip
    logger.info(
        "Built: %s. First failed step: %s", report["success"], report.get("failed_at")
    )


if __name__ == "__main__":
    main()
