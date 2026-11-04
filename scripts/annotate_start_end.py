"""Draw what the system read from a video on its first and last frames.

Left, the first frame: the starting glasses (green, with when each left), where Claude
pointed (blue crosses, if Claude was used), and bright things thrown away as clutter (red).
Right, the last frame: where each glass ended up, numbered in the order it was placed.

PYTHONPATH=. .venv/bin/python scripts/annotate_start_end.py --video data/raw/task2_triangle_clutter_take2.mov
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts.glass_from_video import (
    CONFIG_PATH,
    VideoReadError,
    _background,
    _video_view,
    read_video,
    rough_corners,
)
from src.perception.glass_tracker import (
    GlassVideoConfig,
    find_tops,
    rest_spots,
    search_masks,
)
from src.perception.table_view import refine_corners
from src.perception.vlm_rescue import drop_clutter
from src.simulation.scene import ROOT

POINTS = ROOT / "data/processed/task2_setups/claude_points.json"
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, BLUE, RED, WHITE = (60, 200, 60), (230, 140, 30), (60, 60, 230), (255, 255, 255)
PANEL_WIDTH = 960


def _frame(video: Path, last: bool) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, count - 5 if last else 0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise VideoReadError(f"Couldn't read a frame of {video.name}")
    return frame


def _label(image: np.ndarray, point, text: str, colour) -> None:
    x, y = int(point[0]), int(point[1])
    cv2.circle(image, (x, y), 34, colour, 4, cv2.LINE_AA)
    cv2.putText(image, text, (x + 38, y + 10), FONT, 1.0, WHITE, 6, cv2.LINE_AA)
    cv2.putText(image, text, (x + 38, y + 10), FONT, 1.0, colour, 2, cv2.LINE_AA)


def _cross(image: np.ndarray, point, colour) -> None:
    cv2.drawMarker(image, (int(point[0]), int(point[1])), colour, cv2.MARKER_CROSS, 40, 4)


def _title(image: np.ndarray, text: str) -> np.ndarray:
    scale = PANEL_WIDTH / image.shape[1]
    small = cv2.resize(image, (PANEL_WIDTH, 2 * round(image.shape[0] * scale / 2)))
    cv2.rectangle(small, (0, 0), (PANEL_WIDTH, 40), (25, 25, 25), -1)
    cv2.putText(small, text, (12, 28), FONT, 0.75, WHITE, 2, cv2.LINE_AA)
    return small


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--robot", type=Path, help="the robot's video, to show its last frame")
    parser.add_argument("--note", default="", help="a line to print on the robot's frame")
    args = parser.parse_args()
    config = GlassVideoConfig(**yaml.safe_load(CONFIG_PATH.read_text())["real_video"])
    rescue = yaml.safe_load(CONFIG_PATH.read_text())["vlm_rescue"]
    saved = json.loads(POINTS.read_text()) if POINTS.exists() else {}
    points = [tuple(p) for p in saved.get(args.video.stem, [])]
    finals = [tuple(p) for p in saved.get(f"{args.video.stem}@end", [])]
    view = _video_view(args.video, config)
    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    corners = refine_corners(_background(capture), rough_corners(args.video))
    table, search = search_masks(size, corners, config.room_above_table)
    tops = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        tops.append(find_tops(frame, table, search, config))
    capture.release()
    spots = rest_spots(tops, fps, view, config)
    kept = drop_clutter(spots, points, fps, rescue["early_seconds"], rescue["near_pixels"])
    dropped = [spot for spot in spots if spot not in kept]
    first, last = _frame(args.video, last=False), _frame(args.video, last=True)
    for point in points:
        _cross(first, point, BLUE)
    for point in finals:
        _cross(last, point, BLUE)
    for spot in dropped:
        _label(first, (spot.x, spot.y), "clutter, ignored", RED)
    try:
        demo, _ = read_video(args.video, config, points or None, finals or None)
        verdict = "read OK"
        for number, (start, when) in enumerate(zip(demo.starts, demo.moved_at, strict=True), 1):
            _label(first, view.pixel(start, config.glass_height), f"glass {number}: left at {when:.1f}s", GREEN)
        for number, end in enumerate(demo.ends, 1):
            height = config.glass_height * (2 if number == 3 else 1)
            name = f"{number}: {'top, worked out' if number == 3 and not demo.top_seen else 'placed'}"
            _label(last, view.pixel(end, height), name, GREEN)
    except VideoReadError as error:
        verdict = f"refused: {error}"
        for spot in kept:
            when = f"{spot.first / fps:.1f}-{spot.last / fps:.1f}s"
            _label(first if spot.first / fps <= 1.0 else last, (spot.x, spot.y), when, GREEN)
    left = _title(first, "START: green = glasses found, blue + = Claude, red = clutter ignored")
    right = _title(last, f"END: where the glasses ended up  ({verdict[:60]})")
    panels = [left, right]
    if args.robot:
        robot = _frame(args.robot, last=True)
        panel = _title(robot, "ROBOT: its last frame")
        if args.note:
            cv2.rectangle(panel, (0, panel.shape[0] - 44), (PANEL_WIDTH, panel.shape[0]), (25, 25, 25), -1)
            cv2.putText(panel, args.note, (12, panel.shape[0] - 14), FONT, 0.7, (80, 80, 255), 2, cv2.LINE_AA)
        panels.append(cv2.resize(panel, (PANEL_WIDTH, left.shape[0])))
    output = ROOT / "data/processed" / args.video.stem / "start_and_end.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.hstack(panels))
    print(output, verdict)


if __name__ == "__main__":
    main()
