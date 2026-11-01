"""Inverse kinematics: find arm joint angles that put the hand at a desired pose."""

from dataclasses import dataclass

import mink
import mujoco
import numpy as np

ARM_JOINTS = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
# Leaning the waist extends reach. Twisting it is set directly by waist_yaw_for_target.
WAIST_LEAN_JOINT = "waist_pitch_joint"
WAIST_TWIST_JOINT = "waist_yaw_joint"
SHOULDER_SIDE_OFFSET = 0.10
# Prefer a slight forward lean, so the waist can't drift to its backward limit over
# many small moves and leave the arm without room.
PREFERRED_LEAN = 0.15
LEAN_POSTURE_COST = 0.05
BASE_POSTURE_COST = 1e-3
# Starting poses. The idle left arm hangs back at the side, because leaning the waist
# would otherwise sweep that hand through the objects on the table.
READY_POSE = {
    "right": (-0.5, -0.1, 0.3, 0.9, 1.2, 0.0, 0.0),
    "left": (0.7, 0.45, 0.0, 1.28, 0.0, 0.0, 0.0),
}
SOLVER = "daqp"
IK_DT = 0.02
IK_MAX_ITERATIONS = 200
POSITION_TOLERANCE = 0.005
ORIENTATION_TOLERANCE = 0.05
GENTLE_JOINT_SPEED = 0.5
GENTLE_DAMPING = 0.05


def palm_down_rotation(side: str = "right", pitch: float = 0.0) -> np.ndarray:
    """Hand pointing forward, palm toward the table, fingers tipped down by `pitch`.

    The finger pads are angled, so a small pitch makes them vertical when gripping and
    keeps a held object upright.
    """
    forward = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    down = np.array([-np.sin(pitch), 0.0, -np.cos(pitch)])
    wrist_y = down if side == "right" else -down
    return np.column_stack((forward, wrist_y, np.cross(forward, wrist_y)))


def waist_yaw_for_target(x: float, y: float, side: str) -> float:
    """Twist the torso so the working shoulder faces the target, as a person would.

    This keeps the arm out in front of its own shoulder, so it never has to reach
    across the body and collide with the torso.
    """
    distance = float(np.hypot(x, y))
    shoulder_angle = np.arcsin(min(SHOULDER_SIDE_OFFSET / distance, 1.0))
    sign = 1.0 if side == "right" else -1.0
    return float(np.arctan2(y, x) + sign * shoulder_angle)


def _dof_indices(model: mujoco.MjModel, side: str) -> list[int]:
    names = [f"{side}_{name}_joint" for name in ARM_JOINTS] + [WAIST_LEAN_JOINT]
    return [int(model.joint(name).dofadr[0]) for name in names]


@dataclass(frozen=True)
class IKResult:
    qpos: np.ndarray
    position_error: float
    orientation_error: float

    @property
    def success(self) -> bool:
        return (
            self.position_error < POSITION_TOLERANCE
            and self.orientation_error < ORIENTATION_TOLERANCE
        )


class ArmIK:
    """Moves one arm and the waist lean, freezing all other joints, using mink."""

    def __init__(self, model: mujoco.MjModel, side: str) -> None:
        self._side = side
        self._twist_qpos = int(model.joint(WAIST_TWIST_JOINT).qposadr[0])
        self._configuration = mink.Configuration(model)
        self._frame = mink.FrameTask(
            frame_name=f"{side}_grasp",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.3,
        )
        self._lean_qpos = int(model.joint(WAIST_LEAN_JOINT).qposadr[0])
        posture_cost = np.full(model.nv, BASE_POSTURE_COST)
        posture_cost[int(model.joint(WAIST_LEAN_JOINT).dofadr[0])] = LEAN_POSTURE_COST
        self._posture = mink.PostureTask(model, cost=posture_cost)
        arm_dofs = set(_dof_indices(model, side))
        frozen = [dof for dof in range(model.nv) if dof not in arm_dofs]
        self._freeze = mink.DofFreezingTask(model, dof_indices=frozen)
        self._limits = [mink.ConfigurationLimit(model)]
        # Near a straight arm, tiny hand motions can demand huge joint motions. Gentle
        # motions cap joint speed and add damping, so that can't throw a held object.
        names = [f"{side}_{name}_joint" for name in ARM_JOINTS] + [WAIST_LEAN_JOINT]
        self._gentle_limits = [
            *self._limits,
            mink.VelocityLimit(model, dict.fromkeys(names, GENTLE_JOINT_SPEED)),
        ]

    def solve(
        self,
        qpos: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
        keep_waist: bool = False,
    ) -> IKResult:
        """Solve for a target hand pose, starting from the joint angles in `qpos`.

        Normally the torso turns to face the target, with a slight lean. `keep_waist` stays
        near the current pose instead, for motions that must stay gentle.
        """
        start = qpos.copy()
        if not keep_waist:
            start[self._twist_qpos] = waist_yaw_for_target(
                float(position[0]), float(position[1]), self._side
            )
        self._configuration.update(start)
        preferred = start.copy()
        if not keep_waist:
            preferred[self._lean_qpos] = PREFERRED_LEAN
        self._posture.set_target(preferred)
        self._frame.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(rotation), position
            )
        )
        tasks = [self._frame, self._posture]
        error = self._frame.compute_error(self._configuration)
        for _ in range(IK_MAX_ITERATIONS):
            error = self._frame.compute_error(self._configuration)
            if (
                np.linalg.norm(error[:3]) < POSITION_TOLERANCE / 5
                and np.linalg.norm(error[3:]) < ORIENTATION_TOLERANCE / 5
            ):
                break
            try:
                velocity = mink.solve_ik(
                    self._configuration,
                    tasks,
                    IK_DT,
                    SOLVER,
                    damping=GENTLE_DAMPING if keep_waist else 1e-3,
                    limits=self._gentle_limits if keep_waist else self._limits,
                    constraints=[self._freeze],
                )
            except mink.NoSolutionFound:
                # Keep the closest pose so far. The error returned below reports the miss.
                break
            self._configuration.integrate_inplace(velocity, IK_DT)
        return IKResult(
            qpos=self._configuration.q.copy(),
            position_error=float(np.linalg.norm(error[:3])),
            orientation_error=float(np.linalg.norm(error[3:])),
        )
