"""Turn right-arm settings into left-arm ones.

The G1's arms are mirror images, so anything the right arm does, the left arm can do on the
other side. A joint that turns about a side-to-side axis (pitch) keeps its angle, and one
that turns about any other axis (roll, yaw) changes sign.
"""

from collections.abc import Mapping

import numpy as np

_MIRROR = np.diag([1.0, -1.0, 1.0])
# Shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw.
_ARM_SIGNS = (1, -1, -1, 1, -1, 1, -1)
# Thumb base, thumb middle, thumb tip, then two joints on each of two fingers.
_FINGER_SIGNS = (1, -1, -1, -1, -1, -1, -1)
_KEEP_SIGN = ("pitch", "elbow")


def mirror_arm_pose(pose: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(sign * angle for sign, angle in zip(_ARM_SIGNS, pose, strict=True))


def mirror_fingers(pose: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(sign * angle for sign, angle in zip(_FINGER_SIGNS, pose, strict=True))


def mirror_joint_targets(targets: Mapping[str, float]) -> dict[str, float]:
    """Mirror named waist and arm joints, moving right-arm names to the left arm."""
    return {
        name.replace("right_", "left_"): (
            angle if any(word in name for word in _KEEP_SIGN) else -angle
        )
        for name, angle in targets.items()
    }


def mirror_point(point: tuple[float, ...]) -> tuple[float, ...]:
    """Swap sides across the robot's centre line. Works for (x, y) and (x, y, z)."""
    return (point[0], -point[1], *point[2:])


def mirror_rotation(rotation: np.ndarray) -> np.ndarray:
    return _MIRROR @ rotation @ _MIRROR
