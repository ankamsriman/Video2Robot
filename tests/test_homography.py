"""Converting camera pixels into table positions using the printed marker."""

import cv2
import numpy as np
import pytest

from src.perception.homography import (
    MarkerConfig,
    MarkerNotFoundError,
    TableMapping,
    estimate_mapping,
    table_corners,
)

CONFIG = MarkerConfig(center=(0.24, -0.12), side=0.10, marker_id=0)
MARKER_PIXELS = 200
WIDTH, HEIGHT = 1280, 720


def _camera_view(warp: np.ndarray) -> np.ndarray:
    """Render the marker on a white page as a tilted camera would see it."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, MARKER_PIXELS)
    page = np.full((600, 600), 255, np.uint8)
    page[200:400, 200:400] = marker
    return cv2.warpPerspective(page, warp, (WIDTH, HEIGHT), borderValue=255)


def _tilted_view() -> np.ndarray:
    source = np.float32([[0, 0], [600, 0], [600, 600], [0, 600]])
    target = np.float32([[400, 150], [860, 170], [1000, 600], [280, 580]])
    return cv2.getPerspectiveTransform(source, target)


def test_table_corners_follow_the_printed_arrow_away_from_the_robot() -> None:
    corners = table_corners(CONFIG)

    top_left, top_right, bottom_right, bottom_left = corners
    assert top_left == pytest.approx([0.29, -0.07])
    assert top_right == pytest.approx([0.29, -0.17])
    assert bottom_right == pytest.approx([0.19, -0.17])
    assert bottom_left == pytest.approx([0.19, -0.07])


def test_marker_center_maps_to_its_table_position() -> None:
    warp = _tilted_view()
    mapping = estimate_mapping([_camera_view(warp)], CONFIG)
    center = cv2.perspectiveTransform(np.float32([[[300, 300]]]), warp)[0, 0]

    assert mapping.to_table(center) == pytest.approx((0.24, -0.12), abs=0.003)


def test_points_beyond_the_marker_map_accurately_on_an_ideal_view() -> None:
    warp = _tilted_view()
    mapping = estimate_mapping([_camera_view(warp)], CONFIG)
    # 100 page pixels is 5 cm, because the 200 pixel marker is 10 cm wide.
    away = cv2.perspectiveTransform(np.float32([[[300, 0]]]), warp)[0, 0]

    assert mapping.to_table(away) == pytest.approx((0.24 + 0.15, -0.12), abs=0.01)


def test_robot_left_is_page_left() -> None:
    warp = _tilted_view()
    mapping = estimate_mapping([_camera_view(warp)], CONFIG)
    page_left = cv2.perspectiveTransform(np.float32([[[100, 300]]]), warp)[0, 0]

    assert mapping.to_table(page_left)[1] > -0.12


def test_missing_marker_gives_a_clear_error() -> None:
    blank = np.full((HEIGHT, WIDTH), 255, np.uint8)

    with pytest.raises(MarkerNotFoundError):
        estimate_mapping([blank], CONFIG)


def test_mapping_uses_the_frames_where_the_marker_is_visible() -> None:
    blank = np.full((HEIGHT, WIDTH), 255, np.uint8)
    warp = _tilted_view()

    mapping = estimate_mapping([blank, _camera_view(warp), blank], CONFIG)

    assert isinstance(mapping, TableMapping)


def test_base_position_moves_from_the_near_edge_to_the_object_center() -> None:
    pixels = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    table = np.float32([[0.5, 0.5], [0.5, 0.4], [0.4, 0.4], [0.4, 0.5]])
    mapping = TableMapping(cv2.getPerspectiveTransform(pixels, table))

    position = mapping.base_position((50.0, 50.0), depth=0.06)

    assert position == pytest.approx((0.45 + 0.03, 0.45), abs=1e-3)
