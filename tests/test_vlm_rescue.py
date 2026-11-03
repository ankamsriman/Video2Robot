"""Adding glasses the reader missed, from points a vision model gives."""

import numpy as np

from src.perception.glass_tracker import Spot
from src.perception.vlm_rescue import (
    departure_frame,
    missing_points,
    rescued_spot,
    seen_from_start,
)


def _spot(x: float, y: float, first: int = 0, last: int = 100) -> Spot:
    return Spot(x, y, first, last, last - first + 1)


def test_a_point_near_a_spot_the_reader_found_is_not_missing() -> None:
    spots = [_spot(100, 100), _spot(400, 300)]
    points = [(103.0, 98.0), (700.0, 500.0)]

    assert missing_points(points, spots, fps=30.0, early_seconds=1.0, near_pixels=30) == [
        (700.0, 500.0)
    ]


def test_a_glass_the_reader_noticed_late_is_not_missing() -> None:
    spots = [_spot(700, 500, first=51, last=90)]

    assert missing_points(
        [(700.0, 500.0)], spots, fps=30.0, early_seconds=1.0, near_pixels=30
    ) == []


def test_a_glass_noticed_late_is_moved_back_to_the_start() -> None:
    spots = [_spot(700, 500, first=51, last=90), _spot(100, 100, first=0, last=200)]

    updated = seen_from_start(
        [(700.0, 500.0)], spots, fps=30.0, early_seconds=1.0, near_pixels=30
    )

    assert updated[0].first == 0
    assert updated[1] == spots[1]


def test_the_glass_leaves_at_the_last_frame_its_patch_still_matches() -> None:
    change = np.concatenate((np.full(60, 2.0), np.full(10, 40.0), np.full(20, 3.0), np.full(90, 35.0)))

    left = departure_frame(change, threshold=15.0, fps=30.0, settle_seconds=1.0)

    assert left is not None and abs(left - 89) <= 8


def test_one_noisy_frame_late_on_does_not_count_as_the_glass_still_there() -> None:
    change = np.concatenate((np.full(180, 10.0), np.full(150, 85.0)))
    change[300] = 5.0

    assert abs(departure_frame(change, threshold=45.0, fps=30.0, settle_seconds=1.0) - 179) <= 8


def test_a_glass_that_never_leaves_has_no_departure() -> None:
    assert departure_frame(np.full(100, 2.0), threshold=15.0, fps=30.0, settle_seconds=1.0) is None


def test_a_rescued_spot_runs_from_the_start_to_its_departure() -> None:
    spot = rescued_spot((700.0, 500.0), 89)

    assert (spot.x, spot.y, spot.first, spot.last) == (700.0, 500.0, 0, 89)
