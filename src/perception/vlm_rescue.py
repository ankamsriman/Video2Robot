"""Add starting glasses the bright-top reader missed, using points from a vision model.

A model points to every glass in an early frame. Any it finds that the reader didn't becomes
a starting spot. When that glass left is measured without the model: a small patch of the
picture at its spot keeps looking like the first frame until the glass is taken away.
"""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from src.perception.glass_tracker import Spot

Point = tuple[float, float]


def missing_points(
    points: Sequence[Point],
    spots: Sequence[Spot],
    fps: float,
    early_seconds: float,
    near_pixels: float,
) -> list[Point]:
    """The model's points with no spot from the reader near them at any time.

    A glass the reader only noticed late was still found, so it isn't missing. See
    `seen_from_start` for that case.
    """
    return [
        point
        for point in points
        if all(np.hypot(point[0] - s.x, point[1] - s.y) > near_pixels for s in spots)
    ]


def drop_clutter(
    spots: Sequence[Spot],
    points: Sequence[Point],
    fps: float,
    early_seconds: float,
    near_pixels: float,
) -> tuple[Spot, ...]:
    """Remove bright things the reader took for glasses, using the model's points.

    The model points to every glass early in the video. A spot far from every point is
    something else, such as an earbud case or a laptop stand, if it was there early on, or if
    it appeared before any glass had left: nothing can have been put down by then. Spots
    that appear later, where glasses are put down, are kept.
    """
    if not points:
        return tuple(spots)

    def at_a_glass(spot: Spot) -> bool:
        return any(
            np.hypot(p[0] - spot.x, p[1] - spot.y) <= near_pixels for p in points
        )

    starts = [
        spot for spot in spots if at_a_glass(spot) and spot.first / fps <= early_seconds
    ]
    first_departure = min((spot.last for spot in starts), default=0)
    return tuple(
        spot
        for spot in spots
        if at_a_glass(spot)
        or (spot.first / fps > early_seconds and spot.first > first_departure)
    )


def seen_from_start(
    points: Sequence[Point],
    spots: Sequence[Spot],
    fps: float,
    early_seconds: float,
    near_pixels: float,
) -> tuple[Spot, ...]:
    """Spots the reader noticed late, at a place the model saw a glass from the start.

    That glass was there all along, so its spot starts at the beginning.
    """
    updated = []
    for spot in spots:
        late = spot.first / fps > early_seconds
        at_point = any(
            np.hypot(p[0] - spot.x, p[1] - spot.y) <= near_pixels for p in points
        )
        updated.append(replace(spot, first=0) if late and at_point else spot)
    return tuple(updated)


def departure_frame(
    change: np.ndarray, threshold: float, fps: float, settle_seconds: float
) -> int | None:
    """The last frame where the patch still matched, if it then stays changed.

    A hand passing in front changes the patch only briefly, so a change counts only if it
    lasts at least `settle_seconds` to the end of the video, or until it matches again. The
    changes are smoothed over half a second first, so one noisy frame can't count as a match.
    """
    window = max(int(fps / 2), 1)
    if len(change) >= window:
        padded = np.pad(change, window // 2, mode="edge")
        change = np.convolve(padded, np.ones(window) / window, mode="valid")[
            : len(change)
        ]
    matched = np.nonzero(change <= threshold)[0]
    if len(matched) == 0:
        return None
    last = int(matched[-1])
    if len(change) - 1 - last < settle_seconds * fps:
        return None
    return last


def rescued_spot(point: Point, departure: int) -> Spot:
    return Spot(float(point[0]), float(point[1]), 0, departure, departure + 1)


def patch_change(
    frames_gray: np.ndarray, point: Point, half: int, reference: np.ndarray
) -> float:
    """Mean difference between a small patch now and the same patch at the start."""
    x, y = int(point[0]), int(point[1])
    patch = frames_gray[max(y - half, 0) : y + half, max(x - half, 0) : x + half]
    return float(np.mean(np.abs(patch.astype(float) - reference)))


def _near(spot: Spot, points: Sequence[Point], near_pixels: float) -> bool:
    return any(np.hypot(p[0] - spot.x, p[1] - spot.y) <= near_pixels for p in points)


def _at_the_end(spots: Sequence[Spot], last_frame: int, fps: float, gap_seconds: float):
    return [spot for spot in spots if spot.last >= last_frame - gap_seconds * fps]


def drop_misread_ends(
    spots: Sequence[Spot],
    end_points: Sequence[Point],
    last_frame: int,
    fps: float,
    gap_seconds: float,
    near_pixels: float,
) -> tuple[Spot, ...]:
    """Remove spots still there at the end that sit far from every glass the model saw then.

    The model points to every glass in a frame near the end. A bright spot there that
    matches none of them is a misread, such as a reflection or some clutter.
    """
    if not end_points:
        return tuple(spots)
    final = _at_the_end(spots, last_frame, fps, gap_seconds)
    return tuple(
        s for s in spots if s not in final or _near(s, end_points, near_pixels)
    )


def unseen_ends(
    spots: Sequence[Spot],
    end_points: Sequence[Point],
    last_frame: int,
    fps: float,
    gap_seconds: float,
    near_pixels: float,
) -> list[Point]:
    """Glasses the model saw at the end with no spot from the reader near them then.

    A top that looks dim in the end's lighting is missed this way, as a stacked top often is.
    A glass the reader saw arrive, even if it lost sight of it before the end, isn't missed.
    """
    final = _at_the_end(spots, last_frame, fps, gap_seconds)
    arrived = [spot for spot in spots if spot.first > gap_seconds * fps]
    seen = [*final, *arrived]
    return [p for p in end_points if not any(_near(s, [p], near_pixels) for s in seen)]


def arrival_frame(
    change: np.ndarray, threshold: float, fps: float, settle_seconds: float
) -> int | None:
    """The first frame from which the patch keeps matching the end, if it does for long enough.

    `change` compares the patch in each frame with the same patch near the end. It's
    smoothed over half a second, so a hand passing by can't count as the glass arriving.
    """
    window = max(int(fps / 2), 1)
    if len(change) >= window:
        padded = np.pad(change, window // 2, mode="edge")
        change = np.convolve(padded, np.ones(window) / window, mode="valid")[
            : len(change)
        ]
    differs = np.nonzero(change > threshold)[0]
    first = 0 if len(differs) == 0 else int(differs[-1]) + 1
    if len(change) - first < settle_seconds * fps:
        return None
    return first


def departure_from_order(
    departures: Sequence[int], arrivals: Sequence[int]
) -> int | None:
    """When the one glass whose leaving wasn't seen must have left, from the order of moves.

    Each glass leaves before it lands. Going through the landings in time, the first one that
    the seen departures can't account for was made by the unseen glass, so it left just
    before then. None if every landing is already accounted for.
    """
    for count, arrival in enumerate(sorted(arrivals), start=1):
        if sum(1 for left in departures if left < arrival) < count:
            return arrival - 1
    return None


def unused_glasses(
    start_points: Sequence[Point],
    end_points: Sequence[Point],
    spots: Sequence[Spot],
    departures: Sequence[int | None],
    last_frame: int,
    fps: float,
    gap_seconds: float,
    near_pixels: float,
) -> list[Point]:
    """Glasses the model saw in the same place at the start and the end, that never left.

    The reader must agree the glass stayed: its spot lasts to the end, or, with no spot (a
    glass the right way up shows no bright top), the picture there never changed. A glass
    that left isn't unused, even if the tower is later built where it stood.
    `departures` gives, for each start point, the frame its picture changed for good.
    """
    final = _at_the_end(spots, last_frame, fps, gap_seconds)
    unused = []
    for point, departure in zip(start_points, departures, strict=True):
        if not any(
            np.hypot(point[0] - q[0], point[1] - q[1]) <= near_pixels
            for q in end_points
        ):
            continue
        nearby = [spot for spot in spots if _near(spot, [point], near_pixels)]
        stayed = any(
            spot in final and spot.first / fps <= gap_seconds for spot in nearby
        )
        if stayed or (not nearby and departure is None):
            unused.append(point)
    return unused


def trust_model_departures(
    spots: Sequence[Spot],
    start_points: Sequence[Point],
    end_points: Sequence[Point],
    departures: Sequence[int | None],
    last_frame: int,
    fps: float,
    gap_seconds: float,
    near_pixels: float,
) -> tuple[tuple[Spot, ...], list[Point]]:
    """End the spot of a glass the model saw at the start but not at the end.

    Something bright left beside a glass, such as a white tube, can keep the reader's spot
    alive after the glass is taken. The model saw no glass there at the end, so it left:
    when its picture changed for good, if that was measured, or else by the order of moves.
    Returns the spots, and the points whose departure is still unknown.
    """
    final = _at_the_end(spots, last_frame, fps, gap_seconds)
    kept: list[Spot] = []
    unknown: list[Point] = []
    for spot in spots:
        point = next(
            (
                (p, d)
                for p, d in zip(start_points, departures, strict=True)
                if np.hypot(p[0] - spot.x, p[1] - spot.y) <= near_pixels
            ),
            None,
        )
        gone = point is not None and not any(
            np.hypot(q[0] - spot.x, q[1] - spot.y) <= near_pixels for q in end_points
        )
        if spot not in final or not gone:
            kept.append(spot)
        elif point[1] is not None:
            kept.append(replace(spot, last=point[1], seen=point[1] - spot.first + 1))
        else:
            unknown.append(point[0])
    return tuple(kept), unknown
