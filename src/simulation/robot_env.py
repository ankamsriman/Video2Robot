"""Run the G1 in real physics: smooth hand motion and finger control via joint motors."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

from src.simulation.ik import (
    ARM_JOINTS,
    READY_POSE,
    WAIST_LEAN_JOINT,
    WAIST_TWIST_JOINT,
    ArmIK,
    palm_down_rotation,
)
from src.simulation.scene import CONFIG_PATH

FINGER_JOINTS = (
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "index_0",
    "index_1",
    "middle_0",
    "middle_1",
)
# Finger angles calibrated so the thumb and fingertips sit directly opposite each other,
# at equal height, the given distance apart. Columns: thumb_1, thumb_2, finger_0, finger_1.
GRIP_WIDTHS = (0.02, 0.03, 0.05, 0.07, 0.085, 0.10)
GRIP_ANGLES = (
    (-0.10, -1.10, 0.90, 0.95),
    (0.00, -1.20, 0.85, 0.90),
    (0.25, -1.45, 0.85, 0.70),
    (0.45, -1.55, 0.95, 0.35),
    (0.60, -1.65, 1.05, 0.00),
    (0.70, -1.65, 1.00, 0.00),
)
ARRIVAL_TOLERANCE = 0.02
CORRECTION_STEPS = 6
CORRECTION_TOLERANCE = 0.003
CORRECTION_GAIN = 0.7
logger = logging.getLogger(__name__)

FrameCallback = Callable[[mujoco.MjData], None]


@dataclass(frozen=True)
class MotionConfig:
    arm: str
    grip_height: float
    hover_height: float
    release_clearance: float
    side_gap: float
    hand_speed: float
    finger_time: float
    settle_time: float
    control_rate_hz: float
    finger_force_limit: float
    hand_pitch: float
    open_width: float
    grip_squeeze: float
    release_margin: float
    home_position: tuple[float, float, float]


def load_motion_config(path: Path = CONFIG_PATH) -> MotionConfig:
    raw = yaml.safe_load(path.read_text())["motion"]
    return MotionConfig(**{**raw, "home_position": tuple(raw["home_position"])})


def hand_pose_for_width(width: float) -> tuple[float, ...]:
    """Return all finger joint angles for a given gap between thumb and fingertips."""
    columns = zip(*GRIP_ANGLES, strict=True)
    thumb_1, thumb_2, finger_0, finger_1 = (
        float(np.interp(width, GRIP_WIDTHS, column)) for column in columns
    )
    return (0.0, thumb_1, thumb_2, finger_0, finger_1, finger_0, finger_1)


def min_jerk(t: float) -> float:
    """Smooth 0-to-1 progress curve that starts and stops gently, like a human reach."""
    clipped = float(np.clip(t, 0.0, 1.0))
    return clipped**3 * (10 - 15 * clipped + 6 * clipped**2)


class RobotEnv:
    """Owns the simulation state and turns motion requests into physics steps."""

    def __init__(
        self,
        model: mujoco.MjModel,
        motion: MotionConfig,
        on_frame: FrameCallback | None = None,
        frame_rate_hz: float = 30.0,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.motion = motion
        self._on_frame = on_frame
        self._frame_interval = 1.0 / frame_rate_hz
        self._next_frame_time = 0.0
        self._ik = ArmIK(model, motion.arm)
        self._rotation = palm_down_rotation(motion.arm, motion.hand_pitch)
        self._actuators = {model.actuator(i).name: i for i in range(model.nu)}
        self._site = model.site(f"{motion.arm}_grasp").id
        self._limit_finger_force()
        self._set_ready_pose()

    def _limit_finger_force(self) -> None:
        """Cap finger torque, so grasps depend on realistic squeezing force."""
        limit = self.motion.finger_force_limit
        for name, index in self._actuators.items():
            if "_hand_" in name:
                self.model.actuator_forcelimited[index] = 1
                self.model.actuator_forcerange[index] = (-limit, limit)

    def _set_ready_pose(self) -> None:
        for side, pose in READY_POSE.items():
            for name, value in zip(ARM_JOINTS, pose, strict=True):
                self._set_joint(f"{side}_{name}_joint", value)
        open_pose = hand_pose_for_width(self.motion.open_width)
        for name, angle in zip(FINGER_JOINTS, open_pose, strict=True):
            self._set_joint(f"{self.motion.arm}_hand_{name}_joint", angle)
        self._start_palm_down()
        mujoco.mj_forward(self.model, self.data)
        self._command = {
            name: float(self.data.qpos[self.model.joint(name).qposadr[0]])
            for name in self._actuators
        }

    def _start_palm_down(self) -> None:
        """Begin in the palm-down pose that motions keep, so the arm never snaps to it."""
        home = np.array(self.motion.home_position)
        result = self._ik.solve(self.data.qpos, home, self._rotation)
        if not result.success:
            raise ValueError(f"Home position {home} is out of the robot's reach")
        self.data.qpos[:] = result.qpos

    def _set_joint(self, joint: str, value: float) -> None:
        self.data.qpos[self.model.joint(joint).qposadr[0]] = value

    @property
    def hand_position(self) -> np.ndarray:
        return self.data.site_xpos[self._site].copy()

    def object_position(self, name: str) -> np.ndarray:
        return self.data.xpos[self.model.body(name).id].copy()

    def advance(self, duration: float) -> None:
        """Step physics while holding current targets, compensating for gravity."""
        steps = round(duration / self.model.opt.timestep)
        for _ in range(steps):
            for name, index in self._actuators.items():
                dof = self.model.joint(name).dofadr[0]
                gain = self.model.actuator_gainprm[index, 0]
                self.data.ctrl[index] = (
                    self._command[name] + self.data.qfrc_bias[dof] / gain
                )
            mujoco.mj_step(self.model, self.data)
            self._emit_frame()

    def _emit_frame(self) -> None:
        if self._on_frame is None or self.data.time < self._next_frame_time:
            return
        self._on_frame(self.data)
        self._next_frame_time = self.data.time + self._frame_interval

    def _commanded_qpos(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        for name, value in self._command.items():
            qpos[self.model.joint(name).qposadr[0]] = value
        return qpos

    def _arm_targets(self, qpos: np.ndarray) -> Mapping[str, float]:
        side = self.motion.arm
        names = [f"{side}_{name}_joint" for name in ARM_JOINTS]
        names += [WAIST_LEAN_JOINT, WAIST_TWIST_JOINT]
        return {name: float(qpos[self.model.joint(name).qposadr[0]]) for name in names}

    def move_hand(self, target: np.ndarray, speed: float | None = None) -> bool:
        """Move the grasp point in a straight line.

        Returns True only if the hand physically arrived, so a blocked or unreachable
        move can never be mistaken for a successful one.
        """
        start = self.hand_position
        distance = float(np.linalg.norm(target - start))
        duration = max(distance / (speed or self.motion.hand_speed), 0.3)
        control_dt = 1.0 / self.motion.control_rate_hz
        waypoints = max(int(duration / control_dt), 1)
        for index in range(1, waypoints + 1):
            progress = min_jerk(index / waypoints)
            waypoint = start + progress * (target - start)
            result = self._ik.solve(self._commanded_qpos(), waypoint, self._rotation)
            self._command = {**self._command, **self._arm_targets(result.qpos)}
            self.advance(control_dt)
        self._correct_hand(target)
        error = float(np.linalg.norm(self.hand_position - target))
        if error >= ARRIVAL_TOLERANCE:
            logger.warning("Hand stopped %.1f cm short of %s", error * 100, target)
            return False
        return True

    def _correct_hand(self, target: np.ndarray) -> None:
        """Measure where the hand really is and nudge it onto the target.

        Body sway and joint sag leave a small error. This feedback removes it, in the
        same way a person adjusts their hand while watching it.
        """
        step_time = self.motion.settle_time / CORRECTION_STEPS
        aim = target.copy()
        for _ in range(CORRECTION_STEPS):
            self.advance(step_time)
            error = target - self.hand_position
            if np.linalg.norm(error) < CORRECTION_TOLERANCE:
                continue
            aim = aim + CORRECTION_GAIN * error
            result = self._ik.solve(self._commanded_qpos(), aim, self._rotation)
            self._command = {**self._command, **self._arm_targets(result.qpos)}

    def set_fingers(self, pose: tuple[float, ...]) -> None:
        """Move the active hand's fingers smoothly to a pose."""
        prefix = f"{self.motion.arm}_hand_"
        names = [f"{prefix}{finger}_joint" for finger in FINGER_JOINTS]
        begin = [self._command[name] for name in names]
        control_dt = 1.0 / self.motion.control_rate_hz
        waypoints = max(int(self.motion.finger_time / control_dt), 1)
        for index in range(1, waypoints + 1):
            progress = min_jerk(index / waypoints)
            updates = {
                name: first + progress * (last - first)
                for name, first, last in zip(names, begin, pose, strict=True)
            }
            self._command = {**self._command, **updates}
            self.advance(control_dt)
