"""Follow identical steel glasses by their bright closed tops, and read the task they did.

Steel has no colour to track, and the glasses look the same. But a glass that stands still
keeps a bright top in one spot. So a move is one spot being left and a new spot appearing.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import cv2
import numpy as np

from src.perception.table_view import TableView

OPEN_KERNEL = np.ones((5, 5), np.uint8)
GLASSES_IN_TRIANGLE = 3
SPOTS_IN_TRIANGLE = 2 * GLASSES_IN_TRIANGLE
GLASSES_IN_FLIP = 4
SPOTS_IN_FLIP = GLASSES_IN_FLIP + GLASSES_IN_TRIANGLE


class VideoReadError(ValueError):
    """The video couldn't be read as a clear task."""


@dataclass(frozen=True)
class GlassVideoConfig:
    table_width: float
    table_depth: float
    glass_height: float
    brighter_than_table: float
    min_brightness: float
    max_saturation: float
    min_area: float
    max_area: float
    min_fill: float
    min_roundness: float
    room_above_table: float
    same_spot_pixels: float
    merge_pixels: float
    run_gap_seconds: float
    min_run_seconds: float
    min_run_density: float
    max_top_offset: float
    max_lower_spacing: float
    opening_radius: float
    max_opening_brightness: float


@dataclass(frozen=True)
class Spot:
    """A place where a glass top was seen, and for how long."""

    x: float
    y: float
    first: int
    last: int
    seen: int


@dataclass(frozen=True)
class TriangleDemo:
    """What the person did, in table positions, in the order they did it."""

    starts: tuple[np.ndarray, ...]
    ends: tuple[np.ndarray, np.ndarray, np.ndarray]
    moved_at: tuple[float, ...]
    top_offset: float | None

    @property
    def top_seen(self) -> bool:
        return self.top_offset is not None

    @property
    def tower_center(self) -> np.ndarray:
        return (self.ends[0] + self.ends[1]) / 2

    @property
    def lower_spacing(self) -> float:
        return float(np.linalg.norm(self.ends[0] - self.ends[1]))


def search_masks(
    image_size: tuple[int, int], corners: np.ndarray, room_above: float
) -> tuple[np.ndarray, np.ndarray]:
    """The table itself, and the area searched, which includes room above its near edge."""
    width, height = image_size
    table = np.zeros((height, width), np.uint8)
    cv2.fillPoly(table, [corners.astype(np.int32)], 255)
    raised = corners.copy()
    raised[:2, 1] -= room_above
    search = np.zeros((height, width), np.uint8)
    cv2.fillPoly(search, [raised.astype(np.int32)], 255)
    return table, search


def find_tops(
    frame: np.ndarray, table: np.ndarray, search: np.ndarray, config: GlassVideoConfig
) -> list[tuple[float, float]]:
    """Find round, bright, colourless patches, compared with the table in this same frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    limit = max(
        float(np.median(hsv[..., 2][table > 0])) + config.brighter_than_table,
        config.min_brightness,
    )
    bright = ((hsv[..., 2] > limit) & (hsv[..., 1] < config.max_saturation)).astype(np.uint8)
    bright = cv2.morphologyEx(bright * 255, cv2.MORPH_OPEN, OPEN_KERNEL)
    contours, _ = cv2.findContours(
        cv2.bitwise_and(bright, search), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    tops = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not config.min_area <= area <= config.max_area or len(contour) < 5:
            continue
        (x, y), (first, second), _ = cv2.fitEllipse(contour)
        ellipse_area = np.pi * first * second / 4
        if ellipse_area == 0:
            continue
        round_enough = min(first, second) / max(first, second) > config.min_roundness
        if area / ellipse_area > config.min_fill and round_enough:
            tops.append((float(x), float(y)))
    return tops


def _visible_runs(frames: np.ndarray, fps: float, config: GlassVideoConfig) -> list[np.ndarray]:
    """Split the frames a top was seen in into runs, keeping the steady, lasting ones."""
    pieces = np.split(frames, np.nonzero(np.diff(frames) > config.run_gap_seconds * fps)[0] + 1)
    return [
        piece
        for piece in pieces
        if (piece[-1] - piece[0]) / fps >= config.min_run_seconds
        and len(piece) / (piece[-1] - piece[0] + 1) >= config.min_run_density
    ]


def _on_table(spot: Spot, view: TableView, config: GlassVideoConfig) -> bool:
    """Whether a top at this pixel could belong to a glass standing on the table."""
    forward, left = view.position((spot.x, spot.y), config.glass_height)
    return 0.0 <= forward <= config.table_depth and 0.0 <= left <= config.table_width


def _merge_close(spots: list[Spot], config: GlassVideoConfig) -> list[Spot]:
    """Treat nearby spots as one glass, since two glasses can't stand that close.

    A glass partly covered by another shows a top that seems to shift a little.
    """
    merged: list[Spot] = []
    for spot in sorted(spots, key=lambda item: item.first):
        for index, kept in enumerate(merged):
            if np.hypot(spot.x - kept.x, spot.y - kept.y) < config.merge_pixels:
                merged[index] = replace(
                    kept,
                    first=min(kept.first, spot.first),
                    last=max(kept.last, spot.last),
                    seen=kept.seen + spot.seen,
                )
                break
        else:
            merged.append(spot)
    return merged


def rest_spots(
    tops_per_frame: Iterable[Sequence[tuple[float, float]]],
    fps: float,
    view: TableView,
    config: GlassVideoConfig,
) -> tuple[Spot, ...]:
    """Find where glasses stood, ignoring reflections and glasses passing through.

    A reflection is off the table when treated as a glass top, or flickers too briefly. A
    carried glass doesn't stay in one spot for long enough to count.
    """
    found: list[dict] = []
    for frame, tops in enumerate(tops_per_frame):
        for x, y in tops:
            for spot in found:
                if np.hypot(x - spot["x"], y - spot["y"]) < config.same_spot_pixels:
                    count = len(spot["frames"]) + 1
                    spot["x"] = (spot["x"] * (count - 1) + x) / count
                    spot["y"] = (spot["y"] * (count - 1) + y) / count
                    spot["frames"].append(frame)
                    break
            else:
                found.append({"x": x, "y": y, "frames": [frame]})
    spots = []
    for spot in found:
        runs = _visible_runs(np.array(spot["frames"]), fps, config)
        if runs:
            spots.append(
                Spot(
                    spot["x"],
                    spot["y"],
                    int(runs[0][0]),
                    int(runs[-1][-1]),
                    sum(len(run) for run in runs),
                )
            )
    return tuple(
        sorted(
            _merge_close([s for s in spots if _on_table(s, view, config)], config),
            key=lambda spot: spot.first,
        )
    )


def _read_moves(
    spots: Sequence[Spot], glasses: int, view: TableView, fps: float, config: GlassVideoConfig
) -> TriangleDemo:
    """Pair each spot a glass left with the next new spot, and check the top glass's place."""
    starts = sorted(spots[:glasses], key=lambda spot: spot.last)
    ends = sorted(spots[glasses:], key=lambda spot: spot.first)
    for start, end in zip(starts, ends, strict=False):
        if start.last > end.first:
            raise VideoReadError("A glass seems to arrive before any glass left its spot")
    height = config.glass_height
    lower = tuple(view.position((s.x, s.y), height) for s in ends[:2])
    top = view.position((ends[2].x, ends[2].y), 2 * height)
    offset = float(np.linalg.norm(top - (lower[0] + lower[1]) / 2))
    if offset > config.max_top_offset:
        raise VideoReadError(
            f"The last glass is {offset * 100:.1f} cm from the middle of the other two, "
            "so it doesn't look like a triangle"
        )
    return TriangleDemo(
        tuple(view.position((s.x, s.y), height) for s in starts),
        (*lower, top),
        tuple(s.last / fps for s in starts),
        offset,
    )


def _read_with_top_unseen(
    spots: Sequence[Spot], view: TableView, fps: float, config: GlassVideoConfig
) -> TriangleDemo | None:
    """Work out a triangle whose top glass was never seen, or None if it isn't clearly one.

    A top seen against the bright floor doesn't stand out, so it can be missed. The third
    glass must be on top if all three starts were left, two glasses stand side by side at
    the end, and no spot sits over those two. A spot that does means a start was missed.
    """
    height = config.glass_height
    latest = max(spot.last for spot in spots)
    near_end = latest - config.run_gap_seconds * fps
    lower = sorted((s for s in spots if s.last >= near_end), key=lambda spot: spot.first)
    starts = sorted((s for s in spots if s.last < near_end), key=lambda spot: spot.last)
    if len(lower) != 2 or len(starts) != GLASSES_IN_TRIANGLE:
        return None
    ends = tuple(view.position((s.x, s.y), height) for s in lower)
    if np.linalg.norm(ends[0] - ends[1]) > config.max_lower_spacing:
        return None
    middle = (ends[0] + ends[1]) / 2
    for spot in starts:
        raised = view.position((spot.x, spot.y), 2 * height)
        if np.linalg.norm(raised - middle) <= config.max_top_offset:
            return None
    for start, end in zip(starts, lower, strict=False):
        if start.last > end.first:
            return None
    return TriangleDemo(
        tuple(view.position((s.x, s.y), height) for s in starts),
        (ends[0], ends[1], middle),
        tuple(s.last / fps for s in starts),
        None,
    )


def read_triangle(
    spots: Sequence[Spot], view: TableView, fps: float, config: GlassVideoConfig
) -> TriangleDemo:
    """Read a video of three glasses built into a triangle."""
    if len(spots) == SPOTS_IN_TRIANGLE - 1:
        demo = _read_with_top_unseen(spots, view, fps, config)
        if demo is not None:
            return demo
    if len(spots) != SPOTS_IN_TRIANGLE:
        raise VideoReadError(
            f"Expected {SPOTS_IN_TRIANGLE} resting spots for a triangle, found {len(spots)}"
        )
    return _read_moves(spots, GLASSES_IN_TRIANGLE, view, fps, config)


def read_flip(
    spots: Sequence[Spot], view: TableView, fps: float, config: GlassVideoConfig
) -> TriangleDemo:
    """Read a video of a triangle, plus a fourth glass that leaves and shows no top again.

    The fourth start is included in `starts`. A flipped glass has no bright top to be seen,
    so it adds a start spot but no end spot. `opening_faces_up` then checks the tower.
    """
    if len(spots) != SPOTS_IN_FLIP:
        raise VideoReadError(
            f"Expected {SPOTS_IN_FLIP} resting spots for a flipped tower, found {len(spots)}"
        )
    return _read_moves(spots, GLASSES_IN_FLIP, view, fps, config)


def opening_faces_up(
    frame: np.ndarray, demo: TriangleDemo, view: TableView, config: GlassVideoConfig
) -> bool:
    """Check that the spot above the tower's third glass is dark, as an opening is.

    Nothing can hide the top of a tower from the phone. A closed top there would be bright,
    and an empty spot would show the bright floor or wall behind it.
    """
    rim_height = 3 * config.glass_height
    center = view.pixel(demo.tower_center, rim_height)
    edge = view.pixel(demo.tower_center + [0.0, config.opening_radius], rim_height)
    width = max(int(np.linalg.norm(edge - center)), 1)
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.ellipse(mask, tuple(center.astype(int)), (width, int(width * 0.7)), 0, 0, 360, 255, -1)
    value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2][mask > 0]
    return value.size > 0 and float(np.median(value)) < config.max_opening_brightness
