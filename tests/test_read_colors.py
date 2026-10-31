"""Suggesting colour ranges from a sample of an object in a video frame."""

import cv2
import numpy as np

from src.perception.color_tracker import ColorSpec, track_colors
from src.perception.read_colors import suggest_spec


def _patch(bgr: tuple[int, int, int]) -> np.ndarray:
    return np.full((40, 40, 3), bgr, np.uint8)


def test_suggested_range_contains_the_sampled_hue() -> None:
    spec = suggest_spec("lime", _patch((40, 200, 40))).spec

    hue = cv2.cvtColor(_patch((40, 200, 40)), cv2.COLOR_BGR2HSV)[0, 0, 0]
    assert spec.hue[0] <= hue <= spec.hue[1]


def test_red_gets_a_range_that_wraps_round() -> None:
    spec = suggest_spec("apple", _patch((5, 5, 220))).spec

    assert spec.hue[0] > spec.hue[1]


def test_suggested_spec_actually_tracks_the_object() -> None:
    frame = np.full((200, 300, 3), 170, np.uint8)
    frame[50:100, 100:150] = (30, 30, 210)
    spec = suggest_spec("apple", frame[60:90, 110:140]).spec

    track = track_colors([frame], [spec], 100, 0.5)["apple"]

    assert not np.isnan(track).any()


def test_a_grey_sample_is_flagged_as_hard_to_track() -> None:
    assert not suggest_spec("tray", _patch((200, 200, 200))).is_colorful
    assert suggest_spec("lime", _patch((40, 200, 40))).is_colorful


def test_spec_type_is_reused() -> None:
    assert isinstance(suggest_spec("lime", _patch((40, 200, 40))).spec, ColorSpec)
