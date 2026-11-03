"""Put the person's video beside the robot's, ending together, with labels.

PYTHONPATH=. .venv/bin/python scripts/side_by_side.py --video data/raw/task2_triangle_right_take1.mov
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from src.simulation.scene import ROOT
from src.teleop.hand_retarget import finger_curls, load_teleop_config

PANEL_HEIGHT = 540
FONT = cv2.FONT_HERSHEY_SIMPLEX
HAND_MODEL = ROOT / "checkpoints/hand_landmarker.task"
BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
    (17, 18), (18, 19), (19, 20),
)  # fmt: skip
BONE_COLOUR = (60, 200, 60)
POINT_COLOUR = (40, 200, 230)


def _hand_landmarker() -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


def _draw_hands(
    frame: np.ndarray, landmarker: vision.HandLandmarker, stamp_ms: int, config
) -> np.ndarray:
    """The frame with each tracked hand's 21 points drawn, and how closed its fingers are."""
    height, width = frame.shape[:2]
    rgb = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    found = landmarker.detect_for_video(rgb, stamp_ms)
    shown = frame.copy()
    biggest, curls = 0.0, None
    for image_hand, world_hand in zip(
        found.hand_landmarks, found.hand_world_landmarks, strict=True
    ):
        points = [(int(p.x * width), int(p.y * height)) for p in image_hand]
        for start, end in BONES:
            cv2.line(shown, points[start], points[end], BONE_COLOUR, 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(shown, point, 3, POINT_COLOUR, -1, cv2.LINE_AA)
        size = float(np.ptp(np.array(points), axis=0).sum())
        if size > biggest:
            world = np.array([[p.x, p.y, p.z] for p in world_hand])
            biggest, curls = size, finger_curls(world, config)
    text = (
        "hand not seen"
        if curls is None
        else "fingers closed: thumb {:.0%}  index {:.0%}  middle {:.0%}".format(*curls)
    )
    cv2.rectangle(shown, (0, height - 30), (width, height), (25, 25, 25), -1)
    cv2.putText(shown, text, (8, height - 9), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return shown


def _labelled(frame: np.ndarray, text: str) -> np.ndarray:
    width = int(frame.shape[1] * PANEL_HEIGHT / frame.shape[0])
    panel = cv2.resize(frame, (width, PANEL_HEIGHT))
    cv2.rectangle(panel, (0, 0), (width, 36), (25, 25, 25), -1)
    cv2.putText(panel, text, (10, 25), FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _read_all(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, None, fx=0.5, fy=0.5))
    capture.release()
    return frames, fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--fitted",
        action="store_true",
        help="use the robot run whose positions were moved to fit, and say so on screen",
    )
    parser.add_argument(
        "--two-arms",
        action="store_true",
        help="use the robot run where each arm served its own side",
    )
    parser.add_argument(
        "--hand-path",
        action="store_true",
        help="use the robot run that copied the hand's path",
    )
    parser.add_argument(
        "--show-hand",
        action="store_true",
        help="draw the tracked hand and how closed its fingers are on the person's video",
    )
    args = parser.parse_args()
    output = ROOT / "data/processed" / args.video.stem
    suffix = "_fitted" if args.fitted else "_two_arms" if args.two_arms else ""
    suffix = f"{suffix}_hand_path" if args.hand_path else suffix
    source = "order from this video, positions moved to fit" if args.fitted else "from this video"
    person, fps = _read_all(args.video)
    robot, robot_fps = _read_all(output / f"robot{suffix}.mp4")
    speed = (len(robot) / robot_fps) / (len(person) / fps)
    robot_label = f"G1 in simulation, {source} ({speed:.1f}x speed)"
    person_label = "Recorded person"
    if args.show_hand:
        config = load_teleop_config(ROOT / "configs/cups_real.yaml")
        with _hand_landmarker() as landmarker:
            person = [
                _draw_hands(frame, landmarker, int(index * 1000 / fps), config)
                for index, frame in enumerate(person)
            ]
        person_label = "Recorded person, hand tracked by MediaPipe (21 points)"
        suffix = f"{suffix}_tracked"
    writer = None
    for index, frame in enumerate(person):
        match = robot[min(int(index * len(robot) / len(person)), len(robot) - 1)]
        joined = np.hstack(
            (_labelled(frame, person_label), _labelled(match, robot_label))
        )
        if writer is None:
            writer = cv2.VideoWriter(
                str(output / f"side_by_side{suffix}.mp4"),
                cv2.VideoWriter.fourcc(*"mp4v"),
                fps,
                (joined.shape[1], joined.shape[0]),
            )
        writer.write(joined)
    if writer is not None:
        writer.release()
    print(output / f"side_by_side{suffix}.mp4")


if __name__ == "__main__":
    main()
