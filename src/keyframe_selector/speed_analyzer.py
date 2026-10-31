"""Find pick and place moments: hands slow down when they grasp or release something."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, savgol_filter

POLYNOMIAL_ORDER = 2
MIN_SMOOTHING_FRAMES = 5


@dataclass(frozen=True)
class KeyframeConfig:
    smoothing_seconds: float
    min_gap_seconds: float
    boundary_seconds: float
    min_prominence: float


@dataclass(frozen=True)
class KeyframeResult:
    frames: tuple[int, ...]
    times: tuple[float, ...]
    speed: np.ndarray


def fill_gaps(positions: np.ndarray) -> np.ndarray:
    """Estimate the hand's position in frames where it wasn't detected."""
    seen = ~np.isnan(positions[:, 0])
    if not seen.any():
        raise ValueError("No hand was detected anywhere in the video")
    frames = np.arange(len(positions))
    return np.column_stack(
        [
            np.interp(frames, frames[seen], positions[seen, axis])
            for axis in range(positions.shape[1])
        ]
    )


def hand_speed(positions: np.ndarray, fps: float) -> np.ndarray:
    """Return speed per frame: distance moved since the last frame, divided by time."""
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1) * fps
    return np.concatenate((steps[:1], steps))


def _smooth(speed: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    window = max(int(seconds * fps) | 1, MIN_SMOOTHING_FRAMES)
    if window >= len(speed):
        return speed
    return np.clip(savgol_filter(speed, window, POLYNOMIAL_ORDER), 0.0, None)


def select_keyframes(
    positions: np.ndarray, fps: float, config: KeyframeConfig
) -> KeyframeResult:
    """Pick the valleys of the speed curve, in frames where the hand was really seen."""
    seen = ~np.isnan(positions[:, 0])
    speed = _smooth(
        hand_speed(fill_gaps(positions), fps), fps, config.smoothing_seconds
    )
    valleys, _ = find_peaks(
        -speed,
        distance=max(int(config.min_gap_seconds * fps), 1),
        prominence=config.min_prominence,
    )
    margin = int(config.boundary_seconds * fps)
    frames = tuple(
        int(frame)
        for frame in valleys
        if margin <= frame < len(speed) - margin and seen[frame]
    )
    return KeyframeResult(frames, tuple(frame / fps for frame in frames), speed)


def moments_found(
    labelled: tuple[float, ...], detected: tuple[float, ...], tolerance: float
) -> int:
    """Count labelled moments matched by a detection, using each detection once."""
    remaining = list(detected)
    found = 0
    for moment in labelled:
        distances = [abs(time - moment) for time in remaining]
        if distances and min(distances) <= tolerance:
            remaining.pop(int(np.argmin(distances)))
            found += 1
    return found
