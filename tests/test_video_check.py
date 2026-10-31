"""Checking a test video before recording the rest."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.perception.color_tracker import ColorSpec
from src.perception.homography import MarkerConfig
from src.perception.video_check import (
    Status,
    VideoFacts,
    WorkArea,
    check_facts,
    gather_facts,
)

AREA = WorkArea(x=(0.26, 0.54), y=(-0.40, 0.15))
GOOD = VideoFacts(
    width=1920,
    height=1080,
    fps=30.0,
    seconds=25.0,
    marker_rate=1.0,
    sharpness=200.0,
    color_rates={"apple": 0.95},
    start_positions={"apple": (0.35, -0.28)},
)


def _status(facts: VideoFacts, topic: str) -> Status:
    return next(item.status for item in check_facts(facts, AREA) if item.topic == topic)


def test_a_good_video_has_no_problems() -> None:
    assert all(item.status is Status.OK for item in check_facts(GOOD, AREA))


def test_a_missing_marker_is_a_problem() -> None:
    assert (
        _status(VideoFacts(**{**GOOD.__dict__, "marker_rate": 0.0}), "Marker")
        is Status.PROBLEM
    )


def test_a_marker_seen_only_sometimes_is_a_warning() -> None:
    assert (
        _status(VideoFacts(**{**GOOD.__dict__, "marker_rate": 0.6}), "Marker")
        is Status.WARNING
    )


def test_a_colour_rarely_found_is_a_problem() -> None:
    facts = VideoFacts(**{**GOOD.__dict__, "color_rates": {"apple": 0.1}})

    assert _status(facts, "Colour: apple") is Status.PROBLEM


def test_an_object_out_of_reach_is_a_problem() -> None:
    facts = VideoFacts(**{**GOOD.__dict__, "start_positions": {"apple": (0.70, 0.0)}})

    assert _status(facts, "Reach: apple") is Status.PROBLEM


def test_low_resolution_is_a_warning() -> None:
    facts = VideoFacts(**{**GOOD.__dict__, "width": 640, "height": 360})

    assert _status(facts, "Resolution") is Status.WARNING


def test_a_blurry_video_is_a_warning() -> None:
    assert (
        _status(VideoFacts(**{**GOOD.__dict__, "sharpness": 5.0}), "Sharpness")
        is Status.WARNING
    )


def test_gather_facts_reports_a_video_with_no_marker(tmp_path: Path) -> None:
    video = tmp_path / "blank.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240)
    )
    for _ in range(20):
        writer.write(np.full((240, 320, 3), 128, np.uint8))
    writer.release()
    spec = ColorSpec("apple", (170, 8), 100, 60)
    marker = MarkerConfig((0.24, -0.12), 0.10, 0)

    facts = gather_facts(video, [spec], {"apple": 0.06}, marker)

    assert facts.marker_rate == 0.0
    assert facts.color_rates["apple"] == 0.0
    assert facts.start_positions == {}
    assert facts.seconds == pytest.approx(2.0, abs=0.2)
