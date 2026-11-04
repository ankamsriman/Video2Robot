"""Make a video that shows what the glass reader sees, so a refused video can be understood.

Each frame shows:
- dark-yellow discs: every bright, round patch the reader takes for a glass top in that frame;
- labelled boxes: the resting spots it settled on, and when each was seen;
- the tracked hand: 21 points, the wrist marked, and how closed the fingers are;
- with --claude: Claude's glasses in the first frame (blue) and last frame (magenta), and
  whether each resting spot agrees with one of them (kept) or not (ignored as clutter).

PYTHONPATH=. .venv/bin/python scripts/explain_reading.py --video data/raw/task2_triangle_clutter_take1.mov
"""

import argparse
from pathlib import Path

import cv2
import imageio
import numpy as np
import yaml

from scripts.glass_from_video import CONFIG_PATH, _background, rough_corners
from scripts.side_by_side import _draw_hands, _hand_landmarker
from src.perception.finger_closure import load_finger_settings
from src.perception.glass_tracker import (
    GlassVideoConfig,
    find_tops,
    rest_spots,
    search_masks,
)
from src.perception.table_view import refine_corners, table_view
from src.simulation.scene import ROOT

FONT = cv2.FONT_HERSHEY_SIMPLEX
TOP_FILL = (0, 150, 205)  # dark yellow (BGR)
TOP_RIM = (0, 70, 110)
SPOT_COLOUR = (0, 0, 175)  # dark red (BGR)
TABLE_COLOUR = (0, 120, 0)  # dark green
START_COLOUR = (160, 50, 0)  # dark blue
END_COLOUR = (120, 0, 120)  # dark purple
IGNORED_COLOUR = (140, 140, 0)  # dark teal: stands out on the dark table and the pale floor
LABEL_SCALE = 1.25
LABEL_THICKNESS = 3
CLAUDE_POINTS = ROOT / "data/processed/task2_setups/claude_points.json"


def _claude_points(stem: str) -> tuple[list, list]:
    """Claude's glass tops in the first and last frames, as cached by the comparison runs."""
    import json

    cached = json.loads(CLAUDE_POINTS.read_text()) if CLAUDE_POINTS.exists() else {}
    return cached.get(stem) or [], cached.get(f"{stem}@end") or []


def _agrees(spot, points: list, near: float) -> bool:
    return any(np.hypot(spot.x - x, spot.y - y) <= near for x, y in points)


def _label(image: np.ndarray, text: str, origin, colour) -> None:
    """Large text with a white outline, readable on the dark table and the pale floor."""
    (width, _), _ = cv2.getTextSize(text, FONT, LABEL_SCALE, LABEL_THICKNESS)
    origin = (int(min(max(origin[0], 4), image.shape[1] - width - 4)), int(max(origin[1], 40)))
    cv2.putText(image, text, origin, FONT, LABEL_SCALE, (255, 255, 255), LABEL_THICKNESS + 5,
                cv2.LINE_AA)
    cv2.putText(image, text, origin, FONT, LABEL_SCALE, colour, LABEL_THICKNESS, cv2.LINE_AA)


def _cross(image: np.ndarray, point, colour, text: str) -> None:
    x, y = int(point[0]), int(point[1])
    cv2.drawMarker(image, (x, y), colour, cv2.MARKER_TILTED_CROSS, 56, 7, cv2.LINE_AA)
    if text:
        _label(image, text, (x + 30, y + 48), colour)
OUTPUT_WIDTH = 1280


def draw_top(image: np.ndarray, point, radius: int = 24) -> None:
    """A filled dark-yellow disc with a darker rim: easy to see on any background."""
    centre = (int(point[0]), int(point[1]))
    layer = image.copy()
    cv2.circle(layer, centre, radius, TOP_FILL, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.75, image, 0.25, 0, dst=image)
    cv2.circle(image, centre, radius, TOP_RIM, 3, cv2.LINE_AA)


def _read_frames(video: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames, fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--claude", action="store_true", help="also show Claude's glasses")
    args = parser.parse_args()
    starts, finals = _claude_points(args.video.stem) if args.claude else ([], [])
    near = yaml.safe_load(CONFIG_PATH.read_text())["vlm_rescue"]["near_pixels"]
    config = GlassVideoConfig(**yaml.safe_load(CONFIG_PATH.read_text())["real_video"])
    capture = cv2.VideoCapture(str(args.video))
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    corners = refine_corners(_background(capture), rough_corners(args.video))
    capture.release()
    view = table_view(corners, config.table_width, config.table_depth, size)
    table, search = search_masks(size, corners, config.room_above_table)
    frames, fps = _read_frames(args.video)
    tops = [find_tops(frame, table, search, config) for frame in frames]
    spots = rest_spots(tops, fps, view, config)
    fingers = load_finger_settings(ROOT / "configs/cups_real.yaml")
    output = ROOT / "data/processed" / args.video.stem / "what_the_reader_saw.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    scale = OUTPUT_WIDTH / size[0]
    with (
        _hand_landmarker() as landmarker,
        imageio.get_writer(output, fps=fps, macro_block_size=1) as writer,
    ):
        for index, frame in enumerate(frames):
            shown = _draw_hands(frame, landmarker, int(index * 1000 / fps), fingers)
            cv2.polylines(shown, [corners.astype(np.int32)], True, TABLE_COLOUR, 4)
            for x, y in tops[index]:
                draw_top(shown, (x, y))
            for point in starts:
                _cross(shown, point, START_COLOUR, "")
            for point in finals:
                _cross(shown, point, END_COLOUR, "")
            for number, spot in enumerate(spots, start=1):
                if spot.first <= index <= spot.last:
                    kept = not args.claude or _agrees(spot, [*starts, *finals], near)
                    colour = SPOT_COLOUR if kept else IGNORED_COLOUR
                    corner = (int(spot.x) - 40, int(spot.y) - 40)
                    cv2.rectangle(shown, corner, (int(spot.x) + 40, int(spot.y) + 40), colour, 5)
                    label = f"{spot.first / fps:.1f}-{spot.last / fps:.1f}s"
                    if args.claude:
                        label = f"{'kept' if kept else 'ignored'} {label}"
                    _label(shown, label, (corner[0], corner[1] - 12), colour)
            # Video files need even sizes.
            small = cv2.resize(shown, (OUTPUT_WIDTH, 2 * round(size[1] * scale / 2)))
            banner = (
                (f"{index / fps:4.1f}s   yellow: bright tops (CV)   red: kept (CV and Claude agree)",
                 "teal: ignored (no Claude glass)   blue / purple X: Claude's glasses at start / end")
                if args.claude
                else (f"{index / fps:4.1f}s   yellow: bright patches taken for glass tops",
                      f"red: resting spots ({len(spots)} found, a triangle needs 6)")
            )
            cv2.rectangle(small, (0, 0), (OUTPUT_WIDTH, 66), (20, 20, 20), -1)
            for row, line in enumerate(banner):
                cv2.putText(small, line, (12, 27 + 30 * row), FONT, 0.75, (255, 255, 255), 2,
                            cv2.LINE_AA)
            writer.append_data(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    print(output)


if __name__ == "__main__":
    main()
