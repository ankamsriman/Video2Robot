"""Put the person's video beside the robot's, ending together, with labels.

PYTHONPATH=. .venv/bin/python scripts/side_by_side.py --video data/raw/task2.mov
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.simulation.scene import ROOT

PANEL_HEIGHT = 540
FONT = cv2.FONT_HERSHEY_SIMPLEX


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
    args = parser.parse_args()
    output = ROOT / "data/processed" / args.video.stem
    person, fps = _read_all(args.video)
    robot, robot_fps = _read_all(output / "robot.mp4")
    speed = (len(robot) / robot_fps) / (len(person) / fps)
    robot_label = f"G1 in simulation, from this video ({speed:.1f}x speed)"
    writer = None
    for index, frame in enumerate(person):
        match = robot[min(int(index * len(robot) / len(person)), len(robot) - 1)]
        joined = np.hstack(
            (_labelled(frame, "Recorded person"), _labelled(match, robot_label))
        )
        if writer is None:
            writer = cv2.VideoWriter(
                str(output / "side_by_side.mp4"),
                cv2.VideoWriter.fourcc(*"mp4v"),
                fps,
                (joined.shape[1], joined.shape[0]),
            )
        writer.write(joined)
    if writer is not None:
        writer.release()
    print(output / "side_by_side.mp4")


if __name__ == "__main__":
    main()
