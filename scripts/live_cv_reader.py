"""Show, live from the webcam, what the CV reader takes for glass tops.

It runs the same bright-top rule as the video reader on every frame: dark-yellow discs are the
patches it would count as glass tops, and the green outline is the table it searches. Only
the CV reader runs; Claude and move timing need a recorded video.

PYTHONPATH=. .venv/bin/python scripts/live_cv_reader.py          # press Q to quit
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts.glass_from_video import CONFIG_PATH
from src.perception.glass_tracker import GlassVideoConfig, find_tops, search_masks
from src.perception.table_view import refine_corners

logger = logging.getLogger(__name__)

CAMERA_SIZE = (1920, 1080)
CORNERS_FROM = Path("data/labels/live_webcam_table.json")
TOP_FILL = (0, 150, 205)  # dark yellow (BGR)
TOP_RIM = (0, 70, 110)
TABLE_COLOUR = (60, 200, 60)
SHOWN_WIDTH = 1280
FONT = cv2.FONT_HERSHEY_SIMPLEX


def table_corners(frame: np.ndarray) -> np.ndarray:
    """The live view's hand-picked corners, sharpened from the table's edges when possible."""
    rough = np.array(json.loads(CORNERS_FROM.read_text())["rough_corners"], dtype=float)
    try:
        return refine_corners(frame, rough)
    except (IndexError, ValueError) as error:
        logger.warning("Couldn't fit the table's edges (%s); using the picked corners", error)
        return rough


def draw_top(image: np.ndarray, point, radius: int = 24) -> None:
    """A filled dark-yellow disc with a darker rim: easy to see on any background."""
    centre = (int(point[0]), int(point[1]))
    layer = image.copy()
    cv2.circle(layer, centre, radius, TOP_FILL, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.75, image, 0.25, 0, dst=image)
    cv2.circle(image, centre, radius, TOP_RIM, 3, cv2.LINE_AA)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="webcam number")
    args = parser.parse_args()
    config = GlassVideoConfig(**yaml.safe_load(CONFIG_PATH.read_text())["real_video"])
    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
    if not capture.isOpened():
        raise SystemExit("Couldn't open the webcam: check it's plugged in and allowed")
    time.sleep(1.0)  # let the camera settle its exposure before fitting the table
    ok, frame = capture.read()
    if not ok:
        raise SystemExit("The webcam gave no picture")
    size = (frame.shape[1], frame.shape[0])
    corners = table_corners(frame)
    table, search = search_masks(size, corners, config.room_above_table)
    shown_size = (SHOWN_WIDTH, round(size[1] * SHOWN_WIDTH / size[0]))
    logger.info("Live CV reader running: press Q in the window to quit")
    while True:
        ok, frame = capture.read()
        if not ok:
            logger.warning("Lost the webcam picture")
            break
        tops = find_tops(frame, table, search, config)
        cv2.polylines(frame, [corners.astype(np.int32)], True, TABLE_COLOUR, 3)
        for x, y in tops:
            draw_top(frame, (x, y))
        small = cv2.resize(frame, shown_size)
        cv2.rectangle(small, (0, 0), (SHOWN_WIDTH, 36), (25, 25, 25), -1)
        banner = (
            f"CV reader, live: {len(tops)} bright top(s) taken for glasses (yellow)"
            "   green: table searched   Q: quit"
        )
        cv2.putText(small, banner, (10, 25), FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Live CV reader", small)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
