"""When live control starts, stops and starts over."""

import numpy as np

from scripts.live_teleop import HandFollower, HandReading, _robot
from src.simulation.ik import WAIST_LEAN_JOINT, WAIST_TWIST_JOINT, palm_down_rotation
from tests.test_hand_retarget import CONFIG, HOME


def _reading(x: float = 0.5, y: float = 0.5) -> HandReading:
    return HandReading(
        pixels=np.zeros((21, 2)),
        wrist=np.array([x, y]),
        scale=100.0,
        frame=np.eye(3),
        curls=(0.0, 0.0, 0.0),
    )


def _follower() -> HandFollower:
    return HandFollower(CONFIG, HOME, np.eye(3))


def test_a_still_hand_starts_control() -> None:
    follower = _follower()

    for _ in range(CONFIG.calibration_frames):
        follower.update(_reading())

    assert follower.calibration is not None


def test_a_moving_hand_does_not_start_control() -> None:
    # Walking into the picture: the hand is seen, but keeps moving across it.
    follower = _follower()

    for frame in range(3 * CONFIG.calibration_frames):
        follower.update(_reading(x=0.9 - 0.02 * frame))

    assert follower.calibration is None


def test_control_starts_once_the_hand_settles() -> None:
    follower = _follower()
    for frame in range(10):
        follower.update(_reading(x=0.9 - 0.05 * frame))

    for _ in range(CONFIG.calibration_frames):
        follower.update(_reading(x=0.45))

    assert follower.calibration is not None
    assert follower.calibration.wrist[0] == 0.45


def test_a_short_loss_keeps_the_starting_pose() -> None:
    follower = _follower()
    for _ in range(CONFIG.calibration_frames):
        follower.update(_reading())

    for _ in range(CONFIG.reset_after_lost - 1):
        follower.update(None)

    assert follower.calibration is not None


def test_a_long_loss_starts_over_from_home() -> None:
    follower = _follower()
    for _ in range(CONFIG.calibration_frames):
        follower.update(_reading())
    for _ in range(10):
        follower.update(_reading(y=0.2))
    assert follower.position[2] > HOME[2]

    for _ in range(CONFIG.reset_after_lost):
        follower.update(None)

    assert follower.calibration is None
    assert np.allclose(follower.position, HOME)


def test_live_control_moves_only_the_arm_never_the_waist() -> None:
    env, _ = _robot("right")
    waist = {name: env._command[name] for name in (WAIST_LEAN_JOINT, WAIST_TWIST_JOINT)}
    neutral = palm_down_rotation("right", env.motion.hand_pitch)

    for target in ([0.55, 0.3, 0.8], [0.25, -0.4, 1.1], [0.6, 0.0, 0.78]) * 5:
        env.command_hand(np.array(target), neutral, max_step=0.2)

    assert {name: env._command[name] for name in waist} == waist
