"""Track the hand through a video with MediaPipe, and save its full path to disk."""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

LANDMARKS = 21
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HandPath:
    """Landmarks per frame, as image fractions. Frames with no hand hold NaN."""

    landmarks: np.ndarray
    fps: float

    @property
    def centers(self) -> np.ndarray:
        """The hand's centre in image x and y, per frame."""
        return np.nanmean(self.landmarks[:, :, :2], axis=1)


def _largest_hand(result: vision.HandLandmarkerResult) -> np.ndarray | None:
    if not result.hand_landmarks:
        return None
    hands = [
        np.array([[point.x, point.y, point.z] for point in hand])
        for hand in result.hand_landmarks
    ]
    return max(hands, key=lambda hand: float(np.ptp(hand[:, :2], axis=0).sum()))


def track_hand(video: Path, model: Path) -> HandPath:
    """Run hand tracking over each frame of a video."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    frames = []
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            stamp = int(len(frames) * 1000 / fps)
            hand = _largest_hand(landmarker.detect_for_video(image, stamp))
            frames.append(hand if hand is not None else np.full((LANDMARKS, 3), np.nan))
    capture.release()
    if not frames:
        raise ValueError(f"Video has no readable frames: {video}")
    logger.info("Tracked %d frames at %.1f fps", len(frames), fps)
    return HandPath(np.stack(frames), fps)


def track_both_hands(video: Path, model: Path) -> tuple[np.ndarray, float]:
    """Each hand's centre in pixels, per frame, as (frames, 2, 2). NaN where it wasn't seen."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    frames = []
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            found = landmarker.detect_for_video(image, int(len(frames) * 1000 / fps))
            hands = np.full((2, 2), np.nan)
            for index, hand in enumerate(found.hand_landmarks[:2]):
                hands[index] = (
                    np.mean([point.x for point in hand]) * width,
                    np.mean([point.y for point in hand]) * height,
                )
            frames.append(hands)
    capture.release()
    if not frames:
        raise ValueError(f"Video has no readable frames: {video}")
    return np.stack(frames), fps


def save_hand_path(path: HandPath, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, landmarks=path.landmarks, fps=path.fps)


def load_hand_path(source: Path) -> HandPath:
    with np.load(source) as saved:
        return HandPath(saved["landmarks"], float(saved["fps"]))
