"""Find key moments in a video, saving the hand path, a speed plot and the key frames.

PYTHONPATH=. .venv/bin/python scripts/run_keyframes.py --video data/raw/video_01.mp4
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.keyframe_selector.hand_detector import (
    save_hand_path,
    track_hand,
)
from src.keyframe_selector.speed_analyzer import (
    KeyframeConfig,
    KeyframeResult,
    moments_found,
    select_keyframes,
)
from src.simulation.scene import CONFIG_PATH, ROOT

MODEL = ROOT / "checkpoints/hand_landmarker.task"
LABEL_TOLERANCE_SECONDS = 0.5
logger = logging.getLogger(__name__)


def _save_plot(result: KeyframeResult, fps: float, output: Path) -> None:
    times = np.arange(len(result.speed)) / fps
    figure, axis = plt.subplots(figsize=(10, 3.5))
    axis.plot(times, result.speed, color="#1f6f8b", label="Hand speed")
    axis.scatter(
        result.times,
        result.speed[list(result.frames)],
        color="#c0392b",
        zorder=3,
        label="Key moments",
    )
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Speed (image widths per second)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=120)
    plt.close(figure)


def _save_frames(video: Path, frames: tuple[int, ...], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    for frame in frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if ok:
            cv2.imwrite(str(output / f"frame_{frame:05d}.jpg"), image)
    capture.release()


def _report_against_labels(video: Path, times: tuple[float, ...]) -> None:
    label_path = ROOT / "data/labels" / f"{video.stem}.json"
    if not label_path.exists():
        logger.info("No labels at %s, so accuracy wasn't measured", label_path)
        return
    labelled = tuple(
        moment["time"] for moment in json.loads(label_path.read_text())["moments"]
    )
    found = moments_found(labelled, times, LABEL_TOLERANCE_SECONDS)
    logger.info("Found %d of %d labelled moments", found, len(labelled))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()

    settings = yaml.safe_load(CONFIG_PATH.read_text())["keyframes"]
    output = ROOT / "data/processed" / args.video.stem
    path = track_hand(args.video, MODEL)
    save_hand_path(path, output / "hand_path.npz")
    seen = float(np.mean(~np.isnan(path.landmarks[:, 0, 0])))
    logger.info("Hand seen in %.0f%% of frames", seen * 100)

    result = select_keyframes(path.centers, path.fps, KeyframeConfig(**settings))
    _save_plot(result, path.fps, output / "hand_speed.png")
    _save_frames(args.video, result.frames, output / "keyframes")
    logger.info("Key moments at: %s", [round(time, 2) for time in result.times])
    _report_against_labels(args.video, result.times)


if __name__ == "__main__":
    main()
