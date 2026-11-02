"""Draw where glasses and towers can go on a real video frame, for both tasks.

The areas are those measured by scripts/area_test.py.

PYTHONPATH=. .venv/bin/python scripts/make_area_pictures.py --video data/raw/task2_triangle_right_take1.mov
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from scripts.area_test import (
    FLIP_FOURTH,
    FLIP_STARTS,
    FLIP_TOWER,
    TRIANGLE_STARTS,
    TRIANGLE_TOWER,
    Box,
)
from scripts.glass_from_video import _background
from src.perception.table_view import refine_corners
from src.simulation.scene import ROOT

OUTPUT = ROOT / "docs/recording_views"
ROUGH_CORNERS = [[320, 412], [1255, 412], [1384, 846], [186, 884]]
TABLE_WIDTH, TABLE_DEPTH = 80.0, 50.0
GREEN, ORANGE, BLUE = (60, 170, 40), (0, 140, 255), (230, 100, 30)
WHITE, DARK, YELLOW = (255, 255, 255), (25, 25, 25), (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
GLASS_FREE_AREA = (395, 790, 300, 640)


class TablePicture:
    def __init__(self, video: Path) -> None:
        capture = cv2.VideoCapture(str(video))
        corners = refine_corners(
            _background(capture), np.array(ROUGH_CORNERS, dtype=float)
        )
        flat = np.array(
            [
                [0, 0],
                [0, TABLE_WIDTH],
                [TABLE_DEPTH, TABLE_WIDTH],
                [TABLE_DEPTH, 0],
            ],
            dtype=float,
        )
        self.homography = cv2.findHomography(flat, corners)[0]
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
        _, first = capture.read()
        capture.set(cv2.CAP_PROP_POS_FRAMES, count - 4)
        _, last = capture.read()
        capture.release()
        # The start frame has glasses on the left, where the end frame's table is clear.
        top, bottom, left, right = GLASS_FREE_AREA
        self.blank = first.copy()
        self.blank[top:bottom, left:right] = last[top:bottom, left:right]

    def pixel(self, forward: float, sideways: float) -> np.ndarray:
        point = self.homography @ np.array([forward, TABLE_WIDTH / 2 + sideways, 1.0])
        return point[:2] / point[2]

    def outline(self, box: Box) -> np.ndarray:
        (near, far), (a, b) = box
        corners = ((near, a), (near, b), (far, b), (far, a))
        return np.array([self.pixel(*c) for c in corners]).astype(np.int32)


def _label(image: np.ndarray, text: str, at: tuple[float, float]) -> None:
    (width, height), _ = cv2.getTextSize(text, FONT, 0.6, 1)
    x, y = int(at[0]), int(at[1])
    cv2.rectangle(image, (x - 4, y - height - 5), (x + width + 4, y + 6), DARK, -1)
    cv2.putText(image, text, (x, y), FONT, 0.6, WHITE, 1, cv2.LINE_AA)


def _draw(
    table: TablePicture,
    areas: tuple[tuple[Box, tuple[int, int, int]], ...],
    title: str,
    lines: tuple[str, ...],
    name: str,
) -> None:
    image = table.blank.copy()
    shaded = image.copy()
    for box, color in areas:
        cv2.fillPoly(shaded, [table.outline(box)], color)
    image = cv2.addWeighted(shaded, 0.5, image, 0.5, 0)
    for box, color in areas:
        cv2.polylines(image, [table.outline(box)], True, color, 4)
    middle = (table.pixel(0, 0).astype(int), table.pixel(TABLE_DEPTH, 0).astype(int))
    cv2.line(image, tuple(middle[0]), tuple(middle[1]), WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (image.shape[1], 40), DARK, -1)
    cv2.putText(image, title, (14, 28), FONT, 0.8, WHITE, 2, cv2.LINE_AA)
    for index, line in enumerate(lines):
        _label(image, line, (20, 85 + index * 34))
    start, end = table.pixel(45, 10), table.pixel(45, -24)
    cv2.arrowedLine(
        image, tuple(start.astype(int)), tuple(end.astype(int)), YELLOW, 3, tipLength=0.06
    )
    _label(image, "towards YOUR right", (end[0] + 70, end[1] + 30))
    start, end = table.pixel(5, -36), table.pixel(20, -36)
    cv2.arrowedLine(
        image, tuple(start.astype(int)), tuple(end.astype(int)), YELLOW, 3, tipLength=0.15
    )
    _label(image, "away from you", (end[0] - 150, end[1] + 24))
    cv2.imwrite(str(OUTPUT / f"{name}.png"), image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    table = TablePicture(args.video)
    gaps = (
        "Glasses at least 12 cm apart, centre to centre",
        "At least 15 cm from where the tower will stand",
        "Each glass's BASE goes inside the colour",
    )
    _draw(
        table,
        ((TRIANGLE_STARTS, GREEN),),
        "Task 2 INITIAL: put all 3 glasses inside the GREEN area",
        ("12 to 34 cm from your edge", "Centre line to 30 cm on your right", *gaps),
        "task2_initial_area",
    )
    _draw(
        table,
        ((TRIANGLE_TOWER, BLUE),),
        "Task 2 FINAL: the MIDDLE of the tower goes inside the BLUE area",
        ("24 to 36 cm from your edge", "12 cm on your left to 24 cm on your right"),
        "task2_final_area",
    )
    _draw(
        table,
        ((FLIP_STARTS, GREEN), (FLIP_FOURTH, ORANGE)),
        "Task 3 INITIAL: glasses 1-3 in GREEN, the glass to flip in ORANGE",
        (
            "GREEN: 13 to 19 cm from your edge, 14 cm left to 10 cm right",
            "ORANGE: 23 to 28 cm from your edge, 4 cm left to 10 cm right",
            *gaps,
        ),
        "task3_initial_area",
    )
    _draw(
        table,
        ((FLIP_TOWER, BLUE),),
        "Task 3 FINAL: the MIDDLE of the tower goes inside the BLUE area",
        ("20 to 27 cm from your edge", "26 to 31 cm on your right"),
        "task3_final_area",
    )


if __name__ == "__main__":
    main()
