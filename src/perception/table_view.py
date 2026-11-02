"""Turn pixels into real table positions, using the table's corners and measured size.

Positions are (forward, left) in metres: forward from the person's edge towards the phone,
and left along that edge from the corner on the picture's left.
"""

from dataclasses import dataclass

import cv2
import numpy as np

EDGE_SAMPLES = 120
EDGE_SEARCH_PIXELS = 30
EDGE_MARGIN = 0.08
MIN_EDGE_STEP = 6.0
FIT_ROUNDS = 400
FIT_TOLERANCE_PIXELS = 1.5


@dataclass(frozen=True)
class TableView:
    """Where the camera is, relative to the table."""

    rotation: np.ndarray
    camera: np.ndarray
    lens: np.ndarray
    corners: np.ndarray

    def position(self, pixel: tuple[float, float], height: float) -> np.ndarray:
        """The (forward, left) spot directly below a pixel seen at a height above the table."""
        ray = self.rotation.T @ (np.linalg.inv(self.lens) @ np.array([*pixel, 1.0]))
        side = 1.0 if self.camera[2] > 0 else -1.0
        return (self.camera + (side * height - self.camera[2]) / ray[2] * ray)[:2]

    def pixel(self, place: np.ndarray, height: float) -> np.ndarray:
        """Where a (forward, left) position at a height above the table shows in the picture."""
        side = 1.0 if self.camera[2] > 0 else -1.0
        point = np.array([place[0], place[1], side * height])
        seen = self.lens @ (self.rotation @ (point - self.camera))
        return seen[:2] / seen[2]


def table_view(
    corners: np.ndarray, width: float, depth: float, image_size: tuple[int, int]
) -> TableView:
    """Work out the lens and camera position from four corners and the table's real size.

    Corners run near-left, near-right, far-right, far-left, as seen in the picture.
    """
    flat = np.array([[0, 0], [0, width], [depth, width], [depth, 0]], dtype=float)
    center_x, center_y = image_size[0] / 2, image_size[1] / 2
    homography = cv2.findHomography(flat, corners)[0]
    shift = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1.0]])
    first, second = (shift @ homography)[:, 0], (shift @ homography)[:, 1]
    focal_squared = -(first[:2] @ first[:2] - second[:2] @ second[:2]) / (
        first[2] ** 2 - second[2] ** 2
    )
    if focal_squared <= 0:
        raise ValueError("These corners don't fit a table of the measured size")
    focal = float(np.sqrt(focal_squared))
    lens = np.array([[focal, 0, center_x], [0, focal, center_y], [0, 0, 1.0]])
    points = np.column_stack((flat, np.zeros(4)))
    _, rotation_vector, translation = cv2.solvePnP(
        points, corners, lens, None, flags=cv2.SOLVEPNP_IPPE
    )
    rotation = cv2.Rodrigues(rotation_vector)[0]
    return TableView(rotation, (-rotation.T @ translation).ravel(), lens, corners)


def _edge_points(gray: np.ndarray, start: np.ndarray, end: np.ndarray, inward: np.ndarray):
    along = (end - start) / np.linalg.norm(end - start)
    across = np.array([-along[1], along[0]])
    if across @ inward < 0:
        across = -across
    offsets = np.arange(-EDGE_SEARCH_PIXELS, EDGE_SEARCH_PIXELS + 1)
    height, width = gray.shape
    found = []
    for fraction in np.linspace(EDGE_MARGIN, 1 - EDGE_MARGIN, EDGE_SAMPLES):
        base = start + (end - start) * fraction
        line = base + np.outer(offsets, across)
        columns = np.clip(np.round(line[:, 0]).astype(int), 0, width - 1)
        rows = np.clip(np.round(line[:, 1]).astype(int), 0, height - 1)
        steps = np.diff(gray[rows, columns])
        strongest = int(np.argmin(steps))
        if steps[strongest] < -MIN_EDGE_STEP:
            found.append(base + across * (offsets[strongest] + 0.5))
    return np.array(found)


def _fit_line(points: np.ndarray, flat: bool, rng: np.random.Generator) -> tuple[float, float]:
    """Fit y = a x + b, or x = a y + b for steep sides, ignoring points off the line."""
    run, rise = (points[:, 0], points[:, 1]) if flat else (points[:, 1], points[:, 0])
    best = None
    for _ in range(FIT_ROUNDS):
        i, j = rng.choice(len(points), 2, replace=False)
        if run[i] == run[j]:
            continue
        slope = (rise[j] - rise[i]) / (run[j] - run[i])
        close = abs(slope * run + rise[i] - slope * run[i] - rise) < FIT_TOLERANCE_PIXELS
        if best is None or close.sum() > best.sum():
            best = close
    slope, offset = np.polyfit(run[best], rise[best], 1)
    return float(slope), float(offset)


def refine_corners(image: np.ndarray, rough: np.ndarray) -> np.ndarray:
    """Sharpen hand-picked corners by fitting a line to each of the table's edges.

    Corners are found from the edges because the table's real corners are rounded.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(float)
    middle = rough.mean(axis=0)
    rng = np.random.default_rng(0)
    lines = []
    for index, flat in enumerate((True, False, True, False)):
        start, end = rough[index], rough[(index + 1) % 4]
        points = _edge_points(gray, start, end, middle - (start + end) / 2)
        lines.append((_fit_line(points, flat, rng), flat))
    corners = []
    for index in range(4):
        side_a, side_b = lines[index - 1], lines[index]
        (flat_line, _), (steep_line, _) = (
            (side_b, side_a) if side_b[1] else (side_a, side_b)
        )
        y = (flat_line[0] * steep_line[1] + flat_line[1]) / (1 - flat_line[0] * steep_line[0])
        corners.append((steep_line[0] * y + steep_line[1], y))
    return np.array(corners)
