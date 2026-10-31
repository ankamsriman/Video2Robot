"""Measure which table positions a palm-down hand can reach, and print a map."""

import logging

import mujoco
import numpy as np

from src.simulation.ik import ARM_JOINTS, READY_POSE, ArmIK, palm_down_rotation
from src.simulation.robot_env import load_motion_config
from src.simulation.scene import build_model, load_scene_config

GRIP_HEIGHT_ABOVE_TABLE = 0.04
X_VALUES = np.arange(0.18, 0.56, 0.04)
Y_VALUES = np.arange(-0.50, 0.26, 0.04)
logger = logging.getLogger(__name__)


def ready_qpos(model: mujoco.MjModel) -> np.ndarray:
    data = mujoco.MjData(model)
    for side, pose in READY_POSE.items():
        for name, value in zip(ARM_JOINTS, pose, strict=True):
            data.qpos[model.joint(f"{side}_{name}_joint").qposadr[0]] = value
    return data.qpos.copy()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_scene_config()
    model = build_model(config)
    start = ready_qpos(model)
    z = config.table.top_height + GRIP_HEIGHT_ABOVE_TABLE

    for side in ("right", "left"):
        ik = ArmIK(model, side)
        rotation = palm_down_rotation(side, load_motion_config().hand_pitch)
        logger.info("\n%s hand (# reachable), rows x, columns y:", side)
        logger.info("x\\y   %s", " ".join(f"{y:+.2f}" for y in Y_VALUES))
        for x in X_VALUES[::-1]:
            cells = []
            for y in Y_VALUES:
                y_mirror = y if side == "right" else -y
                result = ik.solve(start, np.array([x, y_mirror, z]), rotation)
                cells.append("  #  " if result.success else "  .  ")
            logger.info("%.2f  %s", x, " ".join(cells))


if __name__ == "__main__":
    main()
