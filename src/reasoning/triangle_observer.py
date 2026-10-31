"""Recognise the cup triangle in a video, and find how to rebuild it.

The marker's conversion to table positions only works on the table, so a raised cup gets
a wrong position. Instead, it's recognised in the image: its base appears above the other
two, and between them. Where to build comes from the two lower cups, which are on the table.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from src.reasoning.demo_observer import (
    ItemInfo,
    ObserverConfig,
    _first_departure,
    _settled_position,
)
from src.verification.triangle import TriangleConfig

MIN_RAISE_FRACTION = 0.2
Position = tuple[float, float]


@dataclass(frozen=True)
class TriangleDemo:
    order: tuple[str, str, str]
    start_positions: Mapping[str, Position]
    center: Position

    def triangle_for(self, base: TriangleConfig) -> TriangleConfig:
        """The triangle settings, built where the video showed it."""
        return replace(base, center=self.center)


def is_stacked_between(top: Position, first: Position, second: Position) -> bool:
    """Check in image pixels: is this cup's base raised, and between the other two?

    It measures along and across the line joining the lower cups, so that it works
    whichever way the camera faces. "Raised" means offset toward the top of the image.
    """
    start, end, point = np.array(first), np.array(second), np.array(top)
    line = end - start
    spacing = float(np.linalg.norm(line))
    along = float((point - start) @ line) / spacing**2
    normal = np.array([line[1], -line[0]]) / spacing
    if normal[1] > 0:
        normal = -normal
    raised = float((point - start) @ normal)
    return 0.0 <= along <= 1.0 and raised >= MIN_RAISE_FRACTION * spacing


def observe_triangle(
    tracks: Mapping[str, np.ndarray],
    pixels: Mapping[str, np.ndarray],
    cups: tuple[ItemInfo, ...],
    fps: float,
    config: ObserverConfig,
) -> TriangleDemo:
    """Find the order cups were placed in, where they started, and where to build."""
    frames = max(int(config.stable_seconds * fps), 1)
    starts = {
        cup.name: _settled_position(tracks[cup.name], frames, cup.name, "start")
        for cup in cups
    }
    departures = {}
    for cup in cups:
        final = _settled_position(tracks[cup.name], frames, cup.name, "end")
        moved = np.linalg.norm(np.array(final) - np.array(starts[cup.name]))
        if moved < config.move_threshold:
            raise ValueError(f"'{cup.name}' didn't move, so this isn't a triangle demo")
        departures[cup.name] = _first_departure(
            tracks[cup.name], starts[cup.name], config.move_threshold
        )
    order = tuple(sorted(departures, key=departures.__getitem__))
    first, second, top = (
        _settled_position(pixels[name], frames, name, "end") for name in order
    )
    if not is_stacked_between(top, first, second):
        raise ValueError(
            f"'{order[2]}' wasn't seen resting between the others: no triangle found"
        )
    lower = [_settled_position(tracks[n], frames, n, "end") for n in order[:2]]
    x, y = np.mean(lower, axis=0)
    return TriangleDemo(order, starts, (float(x), float(y)))
