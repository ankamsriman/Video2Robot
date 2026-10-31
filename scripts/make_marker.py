"""Create the printable A4 page holding the 10 cm ArUco marker used for measuring positions."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/marker.pdf"
DPI = 300
A4_PIXELS = (2480, 3508)
MARKER_CM = 10.0
MARKER_ID = 0
ARROW_LENGTH = 300
TOP_MARGIN = 900


def centimetres_to_pixels(centimetres: float) -> int:
    return round(centimetres / 2.54 * DPI)


def main() -> None:
    side = centimetres_to_pixels(MARKER_CM)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, side)

    page = Image.new("L", A4_PIXELS, 255)
    left = (A4_PIXELS[0] - side) // 2
    page.paste(Image.fromarray(np.asarray(marker)), (left, TOP_MARGIN))

    draw = ImageDraw.Draw(page)
    center_x = A4_PIXELS[0] // 2
    arrow_tip = TOP_MARGIN - 80 - ARROW_LENGTH
    arrow_base = TOP_MARGIN - 80
    draw.line([(center_x, arrow_base), (center_x, arrow_tip)], fill=0, width=14)
    draw.polygon(
        [
            (center_x, arrow_tip - 60),
            (center_x - 45, arrow_tip + 20),
            (center_x + 45, arrow_tip + 20),
        ],
        fill=0,
    )
    font = ImageFont.load_default(size=60)
    text_y = TOP_MARGIN + side + 100
    notes = (
        "Arrow points away from the robot side of the table.",
        f"Print at 100% scale. The black square must measure {MARKER_CM:.0f} cm.",
    )
    for index, note in enumerate(notes):
        draw.text((center_x, text_y + index * 90), note, fill=0, font=font, anchor="ma")

    Image.init()
    page.convert("1").save(OUTPUT, resolution=DPI)


if __name__ == "__main__":
    main()
