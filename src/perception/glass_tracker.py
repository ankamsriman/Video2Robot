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
    min_seen_seconds: float
    min_presence: float
    max_top_offset: float


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

    starts: tuple[np.ndarray, np.ndarray, np.ndarray]
    ends: tuple[np.ndarray, np.ndarray, np.ndarray]
    moved_at: tuple[float, float, float]
    top_offset: float

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


def rest_spots(
    tops_per_frame: Iterable[Sequence[tuple[float, float]]],
    fps: float,
    config: GlassVideoConfig,
) -> tuple[Spot, ...]:
    """Group tops into spots, and keep those seen for long enough and steadily enough.

    Reflections flicker across the table, so they are seen only now and then, while a glass
    stays visible for most of the time it stands somewhere.
    """
    spots: list[Spot] = []
    for frame, tops in enumerate(tops_per_frame):
        for x, y in tops:
            for index, spot in enumerate(spots):
                if np.hypot(x - spot.x, y - spot.y) < config.same_spot_pixels:
                    count = spot.seen + 1
                    spots[index] = replace(
                        spot,
                        x=(spot.x * spot.seen + x) / count,
                        y=(spot.y * spot.seen + y) / count,
                        last=frame,
                        seen=count,
                    )
                    break
            else:
                spots.append(Spot(x, y, frame, frame, 1))
    return tuple(
        sorted(
            (
                spot
                for spot in spots
                if spot.seen >= config.min_seen_seconds * fps
                and spot.seen / (spot.last - spot.first + 1) >= config.min_presence
            ),
            key=lambda spot: spot.first,
        )
    )


def read_triangle(
    spots: Sequence[Spot], view: TableView, fps: float, config: GlassVideoConfig
) -> TriangleDemo:
    """Pair each spot a glass left with the next new spot, and check the top glass's place."""
    if len(spots) != SPOTS_IN_TRIANGLE:
        raise VideoReadError(
            f"Expected {SPOTS_IN_TRIANGLE} resting spots for a triangle, found {len(spots)}"
        )
    starts = sorted(spots[:GLASSES_IN_TRIANGLE], key=lambda spot: spot.last)
    ends = sorted(spots[GLASSES_IN_TRIANGLE:], key=lambda spot: spot.first)
    for start, end in zip(starts, ends, strict=True):
        if start.last > end.first:
            raise VideoReadError("A glass seems to arrive before any glass left its spot")
    height = config.glass_height
    start_places = tuple(view.position((s.x, s.y), height) for s in starts)
    lower = tuple(view.position((s.x, s.y), height) for s in ends[:2])
    top = view.position((ends[2].x, ends[2].y), 2 * height)
    offset = float(np.linalg.norm(top - (lower[0] + lower[1]) / 2))
    if offset > config.max_top_offset:
        raise VideoReadError(
            f"The last glass is {offset * 100:.1f} cm from the middle of the other two, "
            "so it doesn't look like a triangle"
        )
    return TriangleDemo(
        start_places,  # type: ignore[arg-type]
        (*lower, top),
        tuple(s.last / fps for s in starts),  # type: ignore[arg-type]
        offset,
    )
