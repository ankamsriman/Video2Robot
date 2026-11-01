"""Following coloured objects through video frames, without needing a GPU."""

import cv2
import numpy as np
import pytest

from src.perception.color_tracker import ColorSpec, _bottom_middle, track_colors

RED = ColorSpec(name="apple", hue=(170, 10), min_saturation=80, min_value=60)
GREEN = ColorSpec(name="lime", hue=(45, 75), min_saturation=80, min_value=60)
MIN_AREA = 100
AREA_FRACTION = 0.4


def _frame(boxes: list[tuple[int, int, int, tuple[int, int, int]]]) -> np.ndarray:
    """A grey table with squares, given as (left, top, side, BGR colour)."""
    frame = np.full((240, 320, 3), 170, np.uint8)
    for left, top, side, color in boxes:
        cv2.rectangle(frame, (left, top), (left + side, top + side), color, -1)
    return frame


RED_BGR = (30, 30, 210)
GREEN_BGR = (40, 200, 40)


def _track(frames: list[np.ndarray], specs: list[ColorSpec]) -> dict[str, np.ndarray]:
    return track_colors(frames, specs, MIN_AREA, AREA_FRACTION)


def test_reports_the_bottom_middle_of_the_object() -> None:
    tracks = _track([_frame([(100, 50, 40, RED_BGR)])], [RED])

    assert tracks["apple"][0] == pytest.approx((120, 90), abs=2)


def test_hue_range_wraps_around_for_red() -> None:
    pure_red = _frame([(100, 50, 40, (0, 0, 255))])
    wrapped_red = _frame([(100, 50, 40, (40, 0, 255))])

    tracks = _track([pure_red, wrapped_red], [RED])

    assert not np.isnan(tracks["apple"]).any()


def test_tracks_each_colour_separately_as_it_moves() -> None:
    frames = [
        _frame([(50 + step * 10, 50, 40, RED_BGR), (200, 150, 40, GREEN_BGR)])
        for step in range(5)
    ]

    tracks = _track(frames, [RED, GREEN])

    assert tracks["apple"][4][0] - tracks["apple"][0][0] == pytest.approx(40, abs=2)
    assert tracks["lime"][4] == pytest.approx(tracks["lime"][0], abs=1)


def test_a_mostly_covered_object_gives_no_position() -> None:
    whole = _frame([(100, 50, 40, RED_BGR)])
    covered = whole.copy()
    cv2.rectangle(covered, (95, 45), (145, 80), (150, 180, 220), -1)

    tracks = _track([whole, covered], [RED])

    assert np.isnan(tracks["apple"][1]).all()


def test_a_missing_object_gives_no_position() -> None:
    tracks = _track([_frame([])], [RED])

    assert np.isnan(tracks["apple"][0]).all()


def test_small_specks_of_colour_are_ignored() -> None:
    frame = _frame([(100, 50, 40, RED_BGR), (10, 10, 3, RED_BGR)])

    tracks = _track([frame], [RED])

    assert tracks["apple"][0] == pytest.approx((120, 90), abs=2)


def test_a_slightly_tilted_bottom_edge_still_reports_the_edges_middle() -> None:
    blob = np.zeros((100, 100), dtype=np.uint8)
    # A block from columns 20 to 59, whose bottom edge drops 2 rows towards the right.
    for column in range(20, 60):
        blob[30 : 60 + (column - 20) // 20, column] = 1

    x, y = _bottom_middle(blob.astype(bool))

    assert x == pytest.approx(39.5, abs=1.0)
    assert y == pytest.approx(60.0, abs=1.0)


def test_a_thin_gap_does_not_split_an_object_from_its_bottom_edge() -> None:
    # A bright top and a darker front, with a 2-pixel line of glare between them.
    frame = np.full((240, 320, 3), 170, np.uint8)
    cv2.rectangle(frame, (100, 50), (140, 90), RED_BGR, -1)
    cv2.rectangle(frame, (100, 93), (140, 110), (15, 15, 105), -1)
    spec = ColorSpec("apple", (170, 10), 80, 40)

    tracks = _track([frame], [spec])

    assert tracks["apple"][0] == pytest.approx((120, 110), abs=2)
