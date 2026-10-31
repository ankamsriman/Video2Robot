"""Finding pick and place moments from how fast the hand moves."""

import numpy as np
import pytest

from src.keyframe_selector.speed_analyzer import (
    KeyframeConfig,
    fill_gaps,
    hand_speed,
    moments_found,
    select_keyframes,
)

FPS = 30.0
CONFIG = KeyframeConfig(
    smoothing_seconds=0.3,
    min_gap_seconds=0.8,
    boundary_seconds=0.5,
    min_prominence=0.05,
)


def _path_with_pauses(pause_times: list[float], duration: float = 10.0) -> np.ndarray:
    """A hand sweeping back and forth, pausing at the given times."""
    times = np.arange(0, duration, 1 / FPS)
    speed = np.full_like(times, 0.4)
    for pause in pause_times:
        speed *= 1 - np.exp(-(((times - pause) / 0.25) ** 2))
    x = np.cumsum(speed) / FPS
    return np.column_stack((x, np.zeros_like(x)))


def test_speed_of_a_steady_hand_is_distance_over_time() -> None:
    positions = np.column_stack((np.arange(10) * 0.01, np.zeros(10)))

    assert hand_speed(positions, FPS) == pytest.approx(np.full(10, 0.3))


def test_fill_gaps_interpolates_frames_where_the_hand_was_not_seen() -> None:
    positions = np.array([[0.0, 0.0], [np.nan, np.nan], [0.2, 0.4]])

    assert fill_gaps(positions)[1] == pytest.approx([0.1, 0.2])


def test_fill_gaps_rejects_a_video_with_no_hand_at_all() -> None:
    with pytest.raises(ValueError, match="No hand"):
        fill_gaps(np.full((5, 2), np.nan))


def test_keyframes_are_found_at_each_pause() -> None:
    pauses = [2.0, 4.0, 6.0, 8.0]

    result = select_keyframes(_path_with_pauses(pauses), FPS, CONFIG)

    assert len(result.times) == 4
    assert result.times == pytest.approx(pauses, abs=0.15)


def test_pauses_too_close_together_count_once() -> None:
    result = select_keyframes(_path_with_pauses([4.0, 4.3]), FPS, CONFIG)

    assert len(result.times) == 1


def test_stillness_at_the_start_and_end_is_ignored() -> None:
    positions = _path_with_pauses([5.0], duration=10.0)
    positions[:30] = positions[30]
    positions[-30:] = positions[-31]

    result = select_keyframes(positions, FPS, CONFIG)

    assert result.times == pytest.approx([5.0], abs=0.15)


def test_a_gap_where_the_hand_left_the_frame_is_not_a_keyframe() -> None:
    positions = _path_with_pauses([3.0])
    positions[150:210] = np.nan

    result = select_keyframes(positions, FPS, CONFIG)

    assert all(not 5.0 <= time <= 7.0 for time in result.times)


def test_moments_found_counts_each_detection_once() -> None:
    assert moments_found((1.0, 5.0), (1.2, 1.3, 5.1), tolerance=0.5) == 2


def test_moments_found_respects_the_tolerance() -> None:
    assert moments_found((1.0,), (2.0,), tolerance=0.5) == 0
