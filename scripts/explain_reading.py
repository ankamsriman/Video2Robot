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
SPOT_COLOUR = (60, 60, 230)
TABLE_COLOUR = (60, 200, 60)
START_COLOUR = (255, 140, 0)
END_COLOUR = (255, 0, 200)
IGNORED_COLOUR = (150, 150, 150)
CLAUDE_POINTS = ROOT / "data/processed/task2_setups/claude_points.json"


def _claude_points(stem: str) -> tuple[list, list]:
    """Claude's glass tops in the first and last frames, as cached by the comparison runs."""
    import json

    cached = json.loads(CLAUDE_POINTS.read_text()) if CLAUDE_POINTS.exists() else {}
    return cached.get(stem) or [], cached.get(f"{stem}@end") or []


def _agrees(spot, points: list, near: float) -> bool:
    return any(np.hypot(spot.x - x, spot.y - y) <= near for x, y in points)


def _cross(image: np.ndarray, point, colour, text: str) -> None:
    x, y = int(point[0]), int(point[1])
    cv2.drawMarker(image, (x, y), colour, cv2.MARKER_TILTED_CROSS, 46, 4, cv2.LINE_AA)
    cv2.putText(image, text, (x + 26, y + 40), FONT, 0.8, colour, 2, cv2.LINE_AA)
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
            cv2.polylines(shown, [corners.astype(np.int32)], True, TABLE_COLOUR, 2)
            for x, y in tops[index]:
                draw_top(shown, (x, y))
            for point in starts:
                _cross(shown, point, START_COLOUR, "Claude: glass at start")
            for point in finals:
                _cross(shown, point, END_COLOUR, "Claude: glass at end")
            for number, spot in enumerate(spots, start=1):
                if spot.first <= index <= spot.last:
                    kept = not args.claude or _agrees(spot, [*starts, *finals], near)
                    colour = SPOT_COLOUR if kept else IGNORED_COLOUR
                    corner = (int(spot.x) - 40, int(spot.y) - 40)
                    cv2.rectangle(shown, corner, (int(spot.x) + 40, int(spot.y) + 40), colour, 3)
                    label = f"spot {number}: {spot.first / fps:.1f}-{spot.last / fps:.1f}s"
                    if args.claude:
                        label += "  agrees with Claude" if kept else "  no Claude glass: ignored"
                    cv2.putText(
                        shown, label, (corner[0], corner[1] - 8), FONT, 0.8, colour, 2, cv2.LINE_AA
                    )
            # Video files need even sizes.
            small = cv2.resize(shown, (OUTPUT_WIDTH, 2 * round(size[1] * scale / 2)))
            banner = (
                f"{index / fps:4.1f}s  yellow: bright tops (CV)  red: CV spot kept  grey: ignored"
                "  blue/magenta X: Claude's glasses at start/end"
                if args.claude
                else f"{index / fps:4.1f}s   yellow: bright patches taken for glass tops   "
                f"red: resting spots ({len(spots)} found, a triangle needs 6)"
            )
            cv2.rectangle(small, (0, 0), (OUTPUT_WIDTH, 34), (25, 25, 25), -1)
            cv2.putText(small, banner, (10, 23), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            writer.append_data(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    print(output)


if __name__ == "__main__":
    main()
