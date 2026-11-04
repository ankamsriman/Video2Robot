"""How closed a tracked hand's fingers are, for drawing on videos."""

from pathlib import Path

import numpy as np
import pytest

from src.perception.finger_closure import (
    FingerSettings,
    finger_curls,
    load_finger_settings,
)

SETTINGS = FingerSettings(
    finger_bend_open=0.6,
    finger_bend_closed=3.2,
    thumb_reach_open=1.3,
    thumb_reach_closed=0.6,
)


def _hand(curl: float = 0.0, thumb_tucked: bool = False) -> np.ndarray:
    """A made-up right hand in its own frame: x across the palm, y along the fingers."""
    points = np.zeros((21, 3))
    bases = {5: (0.03, 0.09), 9: (0.0, 0.095), 13: (-0.02, 0.09), 17: (-0.04, 0.08)}
    for base, (x, y) in bases.items():
        points[base] = (x, y, 0.0)
        tip = points[base].copy()
        angle = 0.0
        for joint, length in enumerate((0.04, 0.025, 0.02)):
            angle += curl * 1.3
            tip = tip + length * np.array([0.0, np.cos(angle), -np.sin(angle)])
            points[base + 1 + joint] = tip
    points[1] = (0.02, 0.02, 0.0)
    points[2] = (0.045, 0.04, 0.0)
    points[3] = (0.07, 0.05, 0.0)
    points[4] = (0.0, 0.08, -0.02) if thumb_tucked else (0.095, 0.06, 0.0)
    return points - points.mean(axis=0)


def test_an_open_hand_has_straight_fingers() -> None:
    assert finger_curls(_hand(0.0), SETTINGS) == pytest.approx((0.0, 0.0, 0.0), abs=0.05)


def test_a_fist_has_curled_fingers() -> None:
    thumb, index, middle = finger_curls(_hand(1.0, thumb_tucked=True), SETTINGS)

    assert thumb > 0.9
    assert index > 0.9
    assert middle > 0.9


def test_settings_load_from_the_glass_config() -> None:
    settings = load_finger_settings(Path("configs/cups_real.yaml"))

    assert settings.finger_bend_open < settings.finger_bend_closed
    assert settings.thumb_reach_closed < settings.thumb_reach_open
