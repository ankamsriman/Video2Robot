"""Hand tracking's plumbing. Real hands are checked on real video, not here."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.keyframe_selector.hand_detector import (
    HandPath,
    load_hand_path,
    save_hand_path,
    track_hand,
)
from src.simulation.scene import ROOT

MODEL = ROOT / "checkpoints/hand_landmarker.task"


def test_centers_are_the_average_of_the_landmarks() -> None:
    landmarks = np.zeros((1, 21, 3))
    landmarks[0, :, 0] = np.linspace(0.2, 0.4, 21)

    assert HandPath(landmarks, 30.0).centers[0] == pytest.approx([0.3, 0.0])


def test_saved_path_loads_back_unchanged(tmp_path: Path) -> None:
    path = HandPath(np.random.default_rng(0).random((5, 21, 3)), 30.0)
    target = tmp_path / "run" / "hand_path.npz"

    save_hand_path(path, target)
    loaded = load_hand_path(target)

    assert loaded.fps == 30.0
    assert np.array_equal(loaded.landmarks, path.landmarks)


def test_missing_video_gives_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Could not open"):
        track_hand(tmp_path / "missing.mp4", MODEL)


@pytest.mark.integration
@pytest.mark.skipif(
    not MODEL.exists(), reason="run scripts/setup.sh for the hand model"
)
def test_a_video_with_no_hand_gives_empty_frames(tmp_path: Path) -> None:
    video = tmp_path / "blank.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240)
    )
    for _ in range(5):
        writer.write(np.full((240, 320, 3), 128, np.uint8))
    writer.release()

    path = track_hand(video, MODEL)

    assert len(path.landmarks) == 5
    assert np.isnan(path.landmarks).all()
