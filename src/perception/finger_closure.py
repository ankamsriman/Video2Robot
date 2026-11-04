"""How closed a tracked hand's fingers are, for drawing on videos.

MediaPipe gives 21 points on the hand in metres. A finger's closure comes from how much it
bends at its knuckles; the thumb's from how near its tip comes to the middle knuckle, in
palm widths. Used by the side-by-side and reader videos to print "fingers closed: ...".
"""

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import yaml

WRIST = 0
INDEX_BASE = 5
MIDDLE_BASE = 9
PINKY_BASE = 17
THUMB_TIP = 4


@dataclass(frozen=True)
class FingerSettings:
    """Bends and reaches that count as fully open and fully closed. Angles in radians."""

    finger_bend_open: float
    finger_bend_closed: float
    thumb_reach_open: float
    thumb_reach_closed: float


def load_finger_settings(path: Path) -> FingerSettings:
    return FingerSettings(**yaml.safe_load(path.read_text())["finger_closure"])


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _bend(world: np.ndarray, base: int) -> float:
    """Total bend along a finger, in radians: about zero when straight."""
    chain = world[[WRIST, base, base + 1, base + 2, base + 3]]
    bones = [_unit(bone) for bone in np.diff(chain, axis=0)]
    return float(sum(np.arccos(np.clip(a @ b, -1.0, 1.0)) for a, b in pairwise(bones)))


def _fraction(value: float, start: float, end: float) -> float:
    return float(np.clip((value - start) / (end - start), 0.0, 1.0))


def finger_curls(
    world: np.ndarray, settings: FingerSettings
) -> tuple[float, float, float]:
    """How closed the thumb, index and middle finger are, each from 0 (open) to 1 (closed)."""
    palm_width = float(np.linalg.norm(world[INDEX_BASE] - world[PINKY_BASE]))
    reach = float(np.linalg.norm(world[THUMB_TIP] - world[MIDDLE_BASE])) / palm_width
    thumb = _fraction(reach, settings.thumb_reach_open, settings.thumb_reach_closed)
    index, middle = (
        _fraction(
            _bend(world, base), settings.finger_bend_open, settings.finger_bend_closed
        )
        for base in (INDEX_BASE, MIDDLE_BASE)
    )
    return thumb, index, middle
