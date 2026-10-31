"""Run the plan live in the MuJoCo window, at real-time speed.

On a Mac, run it with:
PYTHONPATH=. .venv/bin/mjpython scripts/watch_live.py
"""

import argparse
import logging
import time

import mujoco
import mujoco.viewer

from scripts.phase1_gate import randomised_positions
from src.schemas.plan import load_plan
from src.simulation.primitives import execute_plan
from src.simulation.recorder import CAMERA
from src.simulation.robot_env import RobotEnv, load_motion_config
from src.simulation.scene import ROOT, build_model, load_scene_config

PASSING_SEED = 500
VIEW_RATE_HZ = 60.0
IDLE_SLEEP = 0.02
logger = logging.getLogger(__name__)


class LiveView:
    """Keeps the window in step with the simulation, at real-time speed."""

    def __init__(self) -> None:
        self.viewer: mujoco.viewer.Handle | None = None
        self._start = time.monotonic()

    def on_frame(self, data: mujoco.MjData) -> None:
        if self.viewer is None or not self.viewer.is_running():
            return
        self.viewer.sync()
        ahead = data.time - (time.monotonic() - self._start)
        if ahead > 0:
            time.sleep(ahead)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=PASSING_SEED)
    args = parser.parse_args()

    config = load_scene_config()
    plan = load_plan(ROOT / "data/labels/sample_plan.json")
    model = build_model(config, randomised_positions(config, args.seed))
    view = LiveView()
    env = RobotEnv(
        model, load_motion_config(), on_frame=view.on_frame, frame_rate_hz=VIEW_RATE_HZ
    )

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = CAMERA["lookat"]
        viewer.cam.distance = CAMERA["distance"]
        viewer.cam.azimuth = CAMERA["azimuth"]
        viewer.cam.elevation = CAMERA["elevation"]
        view.viewer = viewer

        result = execute_plan(env, config, plan)
        logger.info("Result: %s. Close the window to exit.", result.success)
        while viewer.is_running():
            time.sleep(IDLE_SLEEP)


if __name__ == "__main__":
    main()
