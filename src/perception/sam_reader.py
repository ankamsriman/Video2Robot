"""Read a video from SAM 2's glass tracks, made by scripts/sam_track_glasses.py.

SAM 2 follows each glass by its outline, so each track already says which start became
which end, and when the glass left, without relying on a shiny top. The tracks are
turned into the same resting spots the bright-top reader makes, so every later check
(the order of moves, which glass is on top, the triangle's shape) still applies.
Claude's points are used for exact positions where they agree with a track.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from src.perception.glass_tracker import Spot

Point = tuple[float, float]
Track = Sequence[Sequence]


def departure(
    track: Track, fps: float, moved_pixels: float, stay_seconds: float
) -> float | None:
    """When the outline moved away from where it started, or vanished, and stayed so."""
    start = next(((x, y) for _, x, y, _ in track if x is not None), None)
    if start is None:
        return None
    moved = [
        x is None or float(np.hypot(x - start[0], y - start[1])) > moved_pixels
        for _, x, y, _ in track
    ]
    frames = [frame for frame, *_ in track]
    for index, is_moved in enumerate(moved):
        if (
            is_moved
            and all(moved[index:])
            and frames[-1] - frames[index] >= stay_seconds * fps
        ):
            return frames[index] / fps
    return None


def final_point(track: Track, fps: float) -> Point | None:
    """Where the outline sits over the last second."""
    last = track[-1][0]
    tail = [(x, y) for frame, x, y, _ in track if x is not None and frame >= last - fps]
    if not tail:
        return None
    return float(np.median([p[0] for p in tail])), float(
        np.median([p[1] for p in tail])
    )


def _first_point(track: Track) -> Point | None:
    return next(((float(x), float(y)) for _, x, y, _ in track if x is not None), None)


def _snap(point: Point, exact: Sequence[Point], near_pixels: float) -> Point:
    """Claude's nearby point, if one is close enough to be the same glass."""
    best = min(
        exact, key=lambda q: np.hypot(q[0] - point[0], q[1] - point[1]), default=None
    )
    if (
        best is not None
        and np.hypot(best[0] - point[0], best[1] - point[1]) <= near_pixels
    ):
        return (float(best[0]), float(best[1]))
    return point


def spots_from_tracks(
    tracks: Mapping[str, Track],
    claude_starts: Sequence[Point],
    claude_ends: Sequence[Point],
    fps: float,
    moved_pixels: float,
    stay_seconds: float,
    near_pixels: float,
) -> tuple[tuple[Spot, ...], list[Point]]:
    """Starting and ending spots, in the order the glasses left, and the unused glasses.

    Each glass that moved gives a start that ends when it left, and an end that begins
    just after. A glass whose outline never moved took no part.
    """
    moved: list[tuple[int, Point, Point, int]] = []
    unused: list[Point] = []
    for track in tracks.values():
        first = _first_point(track)
        left = departure(track, fps, moved_pixels, stay_seconds)
        end = final_point(track, fps)
        if first is None:
            continue
        if left is None or end is None:
            unused.append(first)
            continue
        moved.append((round(left * fps), first, end, int(track[-1][0])))
    moved.sort(key=lambda item: item[0])
    starts = []
    ends = []
    for left, first, end, last in moved:
        x, y = _snap(first, claude_starts, 2 * near_pixels)
        starts.append(Spot(x, y, 0, left, left + 1))
        ex, ey = _snap(end, claude_ends, 2 * near_pixels)
        ends.append(Spot(ex, ey, left + 1, last, last - left))
    return (*starts, *ends), unused
