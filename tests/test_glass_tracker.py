"""Reading identical glasses from a video, using made-up cameras and pictures."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from src.perception.glass_tracker import (
    GlassVideoConfig,
    Spot,
    VideoReadError,
    find_tops,
    opening_faces_up,
    read_flip,
    read_triangle,
    rest_spots,
    search_masks,
)
from src.perception.table_view import refine_corners, table_view

WIDTH, DEPTH = 0.80, 0.50
SIZE = (1620, 1080)
CONFIG = GlassVideoConfig(
    table_width=WIDTH,
    table_depth=DEPTH,
    glass_height=0.115,
    brighter_than_table=28,
    min_brightness=150,
    max_saturation=45,
    min_area=500,
    max_area=9000,
    min_fill=0.82,
    min_roundness=0.45,
    room_above_table=150,
    same_spot_pixels=16,
    merge_pixels=30,
    run_gap_seconds=1.0,
    min_run_seconds=1.0,
    min_run_density=0.15,
    max_top_offset=0.03,
    opening_radius=0.025,
    max_opening_brightness=110,
)


def _camera() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A camera beyond the table's far edge, above it and looking back and down."""
    lens = np.array([[1900.0, 0, 810], [0, 1900.0, 540], [0, 0, 1]])
    eye = np.array([1.3, 0.44, 0.97])
    aim = np.array([0.25, 0.40, 0.0])
    forward = (aim - eye) / np.linalg.norm(aim - eye)
    right = np.cross(forward, [0.0, 0.0, -1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))
    return lens, rotation, -rotation @ eye


def _project(point: tuple[float, float, float]) -> np.ndarray:
    lens, rotation, translation = _camera()
    seen = lens @ (rotation @ np.array(point) + translation)
    return seen[:2] / seen[2]


def _corners() -> np.ndarray:
    return np.array(
        [
            _project(p)
            for p in ((0, 0, 0), (0, WIDTH, 0), (DEPTH, WIDTH, 0), (DEPTH, 0, 0))
        ]
    )


def test_a_table_position_is_recovered_from_its_pixel() -> None:
    view = table_view(_corners(), WIDTH, DEPTH, SIZE)

    found = view.position(tuple(_project((0.27, 0.45, 0))), 0.0)

    assert np.allclose(found, (0.27, 0.45), atol=0.002)


def test_a_raised_glass_top_is_placed_directly_below_itself() -> None:
    view = table_view(_corners(), WIDTH, DEPTH, SIZE)

    found = view.position(tuple(_project((0.27, 0.45, 0.115))), 0.115)

    assert np.allclose(found, (0.27, 0.45), atol=0.003)


def test_corners_that_cannot_fit_the_table_are_refused() -> None:
    corners = np.array([[300.0, 400], [1200, 410], [1000, 900], [500, 880]])

    with pytest.raises(ValueError, match="measured size"):
        table_view(corners, WIDTH, DEPTH, SIZE)


def test_rough_corners_are_sharpened_from_the_table_edges() -> None:
    corners = _corners()
    image = np.full((SIZE[1], SIZE[0], 3), 240, np.uint8)
    cv2.fillPoly(image, [corners.astype(np.int32)], (60, 50, 50))

    refined = refine_corners(image, corners + [[12, -9], [-10, 8], [9, 11], [-8, -12]])

    assert np.abs(refined - corners).max() < 2.5


def _table_middle() -> tuple[int, int]:
    x, y = _corners().mean(axis=0)
    return int(x), int(y)


def _picture_with_top(center: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    image = np.full((SIZE[1], SIZE[0], 3), 240, np.uint8)
    cv2.fillPoly(image, [_corners().astype(np.int32)], (140, 110, 110))
    cv2.ellipse(image, center, size, 0, 0, 360, (225, 225, 225), -1)
    return image


def test_a_round_bright_top_is_found() -> None:
    table, search = search_masks(SIZE, _corners(), CONFIG.room_above_table)

    middle = _table_middle()

    tops = find_tops(_picture_with_top(middle, (38, 28)), table, search, CONFIG)

    assert len(tops) == 1
    assert np.allclose(tops[0], middle, atol=1.5)


def test_a_thin_bright_reflection_is_not_a_top() -> None:
    table, search = search_masks(SIZE, _corners(), CONFIG.room_above_table)

    tops = find_tops(_picture_with_top(_table_middle(), (90, 6)), table, search, CONFIG)

    assert tops == []


def _middle_pixel() -> tuple[float, float]:
    x, y = _project((0.25, 0.40, CONFIG.glass_height))
    return float(x), float(y)


def test_a_steady_top_is_a_spot_and_a_brief_one_is_not() -> None:
    view = table_view(_corners(), WIDTH, DEPTH, SIZE)
    middle = _middle_pixel()
    passing = (middle[0] + 200, middle[1])
    frames = [[middle] for _ in range(60)]
    for index in range(10, 20):
        frames[index] = [*frames[index], passing]

    spots = rest_spots(frames, 30.0, view, CONFIG)

    assert [(round(s.x), round(s.y)) for s in spots] == [tuple(map(round, middle))]


def test_a_reflection_off_the_table_is_not_a_spot() -> None:
    view = table_view(_corners(), WIDTH, DEPTH, SIZE)
    far_edge = tuple(float(v) for v in _project((DEPTH, WIDTH / 2, 0)))

    assert rest_spots([[far_edge] for _ in range(90)], 30.0, view, CONFIG) == ()


def test_a_partly_covered_top_that_shifts_is_still_one_glass() -> None:
    view = table_view(_corners(), WIDTH, DEPTH, SIZE)
    x, y = _middle_pixel()
    frames = [[(x, y)] for _ in range(60)] + [[(x + 20, y)] for _ in range(60)]

    spots = rest_spots(frames, 30.0, view, CONFIG)

    assert len(spots) == 1
    assert (spots[0].first, spots[0].last) == (0, 119)


def _flip_spots() -> list[Spot]:
    spots = _triangle_spots()
    return [*spots[:3], _spot(0.20, 0.10, CONFIG.glass_height, 0, 800), *spots[3:]]


def test_a_flip_video_has_a_fourth_start_and_no_fourth_end() -> None:
    view = table_view(_corners(), WIDTH, DEPTH, SIZE)
    spots = _flip_spots()
    spots[-1] = replace(spots[-1], last=900)

    demo = read_flip(spots, view, 30.0, CONFIG)

    assert len(demo.starts) == 4
    assert np.allclose(demo.starts[3], (0.20, 0.10), atol=0.004)
    assert np.allclose(demo.tower_center, (0.27, 0.44), atol=0.004)


def _flip_demo():
    view = table_view(_corners(), WIDTH, DEPTH, SIZE)
    return view, read_flip(_flip_spots(), view, 30.0, CONFIG)


def _tower_top_picture(value: int) -> np.ndarray:
    view, demo = _flip_demo()
    center = view.pixel(demo.tower_center, 3 * CONFIG.glass_height)
    image = np.full((SIZE[1], SIZE[0], 3), 240, np.uint8)
    cv2.circle(image, tuple(center.astype(int)), 60, (value, value, value), -1)
    return image


def test_a_dark_tower_top_means_the_opening_faces_up() -> None:
    view, demo = _flip_demo()

    assert opening_faces_up(_tower_top_picture(50), demo, view, CONFIG)


def test_a_bright_tower_top_is_not_an_opening() -> None:
    view, demo = _flip_demo()

    assert not opening_faces_up(_tower_top_picture(225), demo, view, CONFIG)


def test_an_empty_spot_above_the_tower_is_not_an_opening() -> None:
    view, demo = _flip_demo()
    empty = np.full((SIZE[1], SIZE[0], 3), 240, np.uint8)

    assert not opening_faces_up(empty, demo, view, CONFIG)


def _spot(forward: float, left: float, height: float, first: int, last: int) -> Spot:
    x, y = _project((forward, left, height))
    return Spot(float(x), float(y), first, last, last - first + 1)


def _triangle_spots(top_left: float = 0.44) -> list[Spot]:
    h = CONFIG.glass_height
    return [
        _spot(0.17, 0.15, h, 0, 90),
        _spot(0.28, 0.23, h, 0, 380),
        _spot(0.35, 0.14, h, 0, 620),
        _spot(0.27, 0.48, h, 220, 880),
        _spot(0.27, 0.40, h, 500, 880),
        _spot(0.27, top_left, 2 * h, 720, 880),
    ]


def test_moves_are_read_in_the_order_they_happened() -> None:
    view = table_view(_corners(), WIDTH, DEPTH, SIZE)

    demo = read_triangle(_triangle_spots(), view, 30.0, CONFIG)

    assert np.allclose(demo.starts[0], (0.17, 0.15), atol=0.004)
    assert np.allclose(demo.starts[2], (0.35, 0.14), atol=0.004)
    assert np.allclose(demo.ends[0], (0.27, 0.48), atol=0.004)
    assert np.allclose(demo.tower_center, (0.27, 0.44), atol=0.004)
    assert demo.lower_spacing == pytest.approx(0.08, abs=0.005)
    assert demo.moved_at == pytest.approx((3.0, 380 / 30, 620 / 30))


def test_a_last_glass_away_from_the_other_two_is_refused() -> None:
    view = table_view(_corners(), WIDTH, DEPTH, SIZE)

    with pytest.raises(VideoReadError, match="triangle"):
        read_triangle(_triangle_spots(top_left=0.60), view, 30.0, CONFIG)


def test_the_wrong_number_of_spots_is_refused() -> None:
    view = table_view(_corners(), WIDTH, DEPTH, SIZE)

    with pytest.raises(VideoReadError, match="Expected 6"):
        read_triangle(_triangle_spots()[:5], view, 30.0, CONFIG)
