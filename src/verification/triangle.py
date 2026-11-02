"""What counts as a real cup triangle, judged only by where the cups end up.

Two cups stand side by side on the table, and a third rests across their tops.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

Vec3 = tuple[float, float, float]
HEIGHT_TOLERANCE = 0.015


@dataclass(frozen=True)
class TriangleConfig:
    center: tuple[float, float]
    cup_spacing: float
    cup_height: float
    rim_radius: float
    top_radius: float
    position_tolerance: float
    max_tilt_degrees: float
    # The hand passes over the side it fills first, so the arm's far side goes first.
    first_on_left: bool = True


@dataclass(frozen=True)
class CupPose:
    position: Vec3
    up: Vec3


def load_triangle_config(path: Path) -> TriangleConfig:
    raw = yaml.safe_load(path.read_text())["triangle"]
    return TriangleConfig(**{**raw, "center": tuple(raw["center"])})


def triangle_targets(config: TriangleConfig, table: float) -> tuple[Vec3, Vec3, Vec3]:
    """Return where the centres of the left, right and top cups should be."""
    x, y = config.center
    half_gap = config.cup_spacing / 2 if config.first_on_left else -config.cup_spacing / 2
    lower = table + config.cup_height / 2
    return (
        (x, y + half_gap, lower),
        (x, y - half_gap, lower),
        (x, y, lower + config.cup_height),
    )


def _tilt_degrees(pose: CupPose) -> float:
    return float(np.degrees(np.arccos(np.clip(pose.up[2], -1.0, 1.0))))


def triangle_failures(
    left: CupPose, right: CupPose, top: CupPose, config: TriangleConfig, table: float
) -> tuple[str, ...]:
    """List everything wrong with the arrangement. An empty list means success."""
    problems: list[str] = []
    lower_height = table + config.cup_height / 2
    a, b, c = (np.array(pose.position) for pose in (left, right, top))

    for label, pose in (("left", left), ("right", right)):
        if _tilt_degrees(pose) > config.max_tilt_degrees:
            problems.append(f"{label} cup is tilted")
        if abs(pose.position[2] - lower_height) > HEIGHT_TOLERANCE:
            problems.append(f"{label} cup is not standing on the table")
    gap = float(np.linalg.norm(a[:2] - b[:2]))
    if abs(gap - config.cup_spacing) > config.position_tolerance:
        problems.append("lower cups are not side by side")
    if _tilt_degrees(top) > config.max_tilt_degrees:
        problems.append("top cup is tilted")
    if abs(c[2] - (lower_height + config.cup_height)) > HEIGHT_TOLERANCE:
        problems.append("top cup is not resting on the lower cups")
    if float(np.linalg.norm(c[:2] - (a[:2] + b[:2]) / 2)) > config.position_tolerance:
        problems.append("top cup is not centred over both")
    return tuple(problems)
