"""Move a demonstrated layout just enough to fit where the robot's flip works.

The arm's flip only works in part of the table. If a video was recorded elsewhere, each
position moves to the nearest place the robot can use. The order and the flip still come
from the video, and every shift is reported, so that the result stays honest.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SEARCH_STEP = 0.01


@dataclass(frozen=True)
class Area:
    """A rectangle on the table, in metres, as (low, high) along x and along y."""

    x: tuple[float, float]
    y: tuple[float, float]

    def nearest(self, point: np.ndarray) -> np.ndarray:
        return np.array(
            [np.clip(point[0], *self.x), np.clip(point[1], *self.y)], dtype=float
        )


@dataclass(frozen=True)
class FittedLayout:
    starts: tuple[np.ndarray, ...]
    tower: np.ndarray
    start_shifts: tuple[float, ...]
    tower_shift: float


def _nearest_free(
    wanted: np.ndarray, area: Area, keep_clear: Sequence[tuple[np.ndarray, float]]
) -> np.ndarray:
    """The closest spot to `wanted` in the area, staying `gap` from each listed point."""
    xs = np.arange(area.x[0], area.x[1] + SEARCH_STEP / 2, SEARCH_STEP)
    ys = np.arange(area.y[0], area.y[1] + SEARCH_STEP / 2, SEARCH_STEP)
    grid = np.array([(x, y) for x in xs for y in ys])
    free = np.ones(len(grid), dtype=bool)
    for point, gap in keep_clear:
        free &= np.linalg.norm(grid - point, axis=1) >= gap
    if not free.any():
        raise ValueError("No free spot is left in the robot's working area")
    usable = grid[free]
    return usable[np.argmin(np.linalg.norm(usable - wanted, axis=1))]


def fit_flip_layout(
    starts: Sequence[np.ndarray],
    tower: np.ndarray,
    start_area: Area,
    flip_area: Area,
    tower_area: Area,
    glass_gap: float,
    tower_gap: float,
) -> FittedLayout:
    """Fit the tower, then the glass to flip (the last start), then the other glasses."""
    new_tower = tower_area.nearest(tower)
    flipped = _nearest_free(starts[-1], flip_area, [(new_tower, tower_gap)])
    placed: list[np.ndarray] = []
    for start in starts[:-1]:
        keep_clear = [
            (new_tower, tower_gap),
            (flipped, glass_gap),
            *((spot, glass_gap) for spot in placed),
        ]
        placed.append(_nearest_free(start, start_area, keep_clear))
    fitted = (*placed, flipped)
    return FittedLayout(
        fitted,
        new_tower,
        tuple(float(np.linalg.norm(a - b)) for a, b in zip(fitted, starts, strict=True)),
        float(np.linalg.norm(new_tower - tower)),
    )
