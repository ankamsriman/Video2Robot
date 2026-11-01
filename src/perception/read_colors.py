"""Suggest a colour range for tracking, from a sample of an object in a video frame."""

from dataclasses import dataclass

import cv2
import numpy as np

from src.perception.color_tracker import ColorSpec

HUE_MARGIN = 8
HUE_RANGE = 180
MIN_COLORFUL_SATURATION = 60
MINIMUM_SATURATION = 40
MINIMUM_VALUE = 30
# An object's shaded side can be far darker than the lit part the sample comes from. It
# must stay in range, as the tracker finds the object's base from that side's bottom edge.
VALUE_FRACTION = 4


@dataclass(frozen=True)
class SuggestedColor:
    spec: ColorSpec
    is_colorful: bool


def _median_hue(hues: np.ndarray) -> int:
    """Average on a circle, because hue wraps round: red sits at both ends."""
    angles = hues.astype(float) * 2 * np.pi / HUE_RANGE
    mean = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return round(mean * HUE_RANGE / (2 * np.pi)) % HUE_RANGE


def suggest_spec(name: str, sample: np.ndarray) -> SuggestedColor:
    """Build a colour range from a BGR image patch taken from inside the object."""
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue = _median_hue(hsv[:, 0])
    saturation = int(np.median(hsv[:, 1]))
    value = int(np.median(hsv[:, 2]))
    spec = ColorSpec(
        name=name,
        hue=((hue - HUE_MARGIN) % HUE_RANGE, (hue + HUE_MARGIN) % HUE_RANGE),
        min_saturation=max(MINIMUM_SATURATION, saturation // 2),
        min_value=max(MINIMUM_VALUE, value // VALUE_FRACTION),
    )
    return SuggestedColor(spec, saturation >= MIN_COLORFUL_SATURATION)
