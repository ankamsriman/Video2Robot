"""Make a video that shows what the glass reader sees, so a refused video can be understood.

Each frame shows:
- yellow rings: every bright, round patch the reader takes for a glass top in that frame;
- labelled boxes: the resting spots it settled on, and when each was seen;
- the tracked hand: 21 points, the wrist marked, and how closed the fingers are.

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
from src.perception.glass_tracker import (
    GlassVideoConfig,
    find_tops,
    rest_spots,
    search_masks,
)
from src.perception.table_view import refine_corners, table_view
from src.simulation.scene import ROOT
from src.teleop.hand_retarget import load_teleop_config

FONT = cv2.FONT_HERSHEY_SIMPLEX
TOP_COLOUR = (0, 220, 255)
SPOT_COLOUR = (60, 60, 230)
TABLE_COLOUR = (60, 200, 60)
OUTPUT_WIDTH = 1280


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
    args = parser.parse_args()
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
    teleop = load_teleop_config(ROOT / "configs/cups_real.yaml")
    output = ROOT / "data/processed" / args.video.stem / "what_the_reader_saw.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    scale = OUTPUT_WIDTH / size[0]
    with (
        _hand_landmarker() as landmarker,
        imageio.get_writer(output, fps=fps, macro_block_size=1) as writer,
    ):
        for index, frame in enumerate(frames):
            shown = _draw_hands(frame, landmarker, int(index * 1000 / fps), teleop)
            cv2.polylines(shown, [corners.astype(np.int32)], True, TABLE_COLOUR, 2)
            for x, y in tops[index]:
                cv2.circle(shown, (int(x), int(y)), 22, TOP_COLOUR, 3, cv2.LINE_AA)
            for number, spot in enumerate(spots, start=1):
                if spot.first <= index <= spot.last:
                    corner = (int(spot.x) - 40, int(spot.y) - 40)
                    cv2.rectangle(
                        shown, corner, (int(spot.x) + 40, int(spot.y) + 40), SPOT_COLOUR, 3
                    )
                    label = f"spot {number}: {spot.first / fps:.1f}-{spot.last / fps:.1f}s"
                    cv2.putText(
                        shown, label, (corner[0], corner[1] - 8), FONT, 0.8, SPOT_COLOUR, 2, cv2.LINE_AA
                    )
            # Video files need even sizes.
            small = cv2.resize(shown, (OUTPUT_WIDTH, 2 * round(size[1] * scale / 2)))
            banner = (
                f"{index / fps:4.1f}s   yellow: bright patches taken for glass tops   "
                f"red: resting spots ({len(spots)} found, a triangle needs 6)"
            )
            cv2.rectangle(small, (0, 0), (OUTPUT_WIDTH, 34), (25, 25, 25), -1)
            cv2.putText(small, banner, (10, 23), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            writer.append_data(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    print(output)


if __name__ == "__main__":
    main()
