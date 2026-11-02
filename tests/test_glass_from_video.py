"""Choosing the table's rough corners for a video."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.glass_from_video import VideoReadError, rough_corners


def _write(path: Path, corners: list[list[int]]) -> None:
    path.write_text(json.dumps({"rough_corners": corners}))


CORNERS = [[1, 2], [3, 4], [5, 6], [7, 8]]
OTHERS = [[10, 20], [30, 40], [50, 60], [70, 80]]


def test_a_video_with_its_own_corners_uses_them(tmp_path: Path) -> None:
    _write(tmp_path / "take_table.json", CORNERS)
    _write(tmp_path / "current_setup_table.json", OTHERS)

    assert np.array_equal(rough_corners(Path("take.mov"), tmp_path), CORNERS)


def test_a_new_video_uses_the_current_camera_setup(tmp_path: Path) -> None:
    _write(tmp_path / "current_setup_table.json", OTHERS)

    assert np.array_equal(rough_corners(Path("take.mov"), tmp_path), OTHERS)


def test_no_corners_at_all_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(VideoReadError, match="table corners"):
        rough_corners(Path("take.mov"), tmp_path)
