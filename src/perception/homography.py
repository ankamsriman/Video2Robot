"""Convert camera pixels into table positions, using the printed ArUco marker.

This mapping is only correct for points on the table surface, so use the point where an
object touches the table, not its centre.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_UP_STEP = 10.0


class MarkerNotFoundError(ValueError):
    """The marker wasn't visible in any of the frames provided."""


@dataclass(frozen=True)
class MarkerConfig:
    center: tuple[float, float]
    side: float
    marker_id: int


def table_corners(config: MarkerConfig) -> np.ndarray:
    """Table positions of the marker's corners, in the order the detector reports them.

    The order is top-left, top-right, bottom-right, bottom-left as printed. The printed
    arrow points away from the robot (+x), so the page's left is the robot's left (+y).
    """
    x, y = config.center
    half = config.side / 2
    return np.array(
        [
            [x + half, y + half],
            [x + half, y - half],
            [x - half, y - half],
            [x - half, y + half],
        ]
    )


@dataclass(frozen=True)
class TableMapping:
    pixels_to_table: np.ndarray

    def to_table(self, pixel: tuple[float, float]) -> tuple[float, float]:
        point = np.array([[pixel]], dtype=np.float64)
        x, y = cv2.perspectiveTransform(point, self.pixels_to_table)[0, 0]
        return (float(x), float(y))

    def base_position(
        self, near_edge_pixel: tuple[float, float], depth: float
    ) -> tuple[float, float]:
        """Estimate an object's centre on the table from its near edge in the image.

        The camera sees the side of the object nearest to it, so the centre lies half the
        object's depth further away, which is upward in the image.
        """
        near = np.array(self.to_table(near_edge_pixel))
        u, v = near_edge_pixel
        further = np.array(self.to_table((u, v - IMAGE_UP_STEP)))
        direction = (further - near) / np.linalg.norm(further - near)
        x, y = near + direction * depth / 2
        return (float(x), float(y))


def _find_corners(frame: np.ndarray, config: MarkerConfig) -> np.ndarray | None:
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(
        gray
    )
    if ids is None:
        return None
    for found, marker_id in zip(corners, ids.ravel(), strict=True):
        if marker_id == config.marker_id:
            return found[0]
    return None


def estimate_mapping(
    frames: Iterable[np.ndarray], config: MarkerConfig
) -> TableMapping:
    """Build the mapping, using the median corner position across all frames."""
    detections = [
        corners
        for corners in (_find_corners(frame, config) for frame in frames)
        if corners is not None
    ]
    if not detections:
        raise MarkerNotFoundError(
            "The printed marker wasn't found. Check it's fully visible and in focus."
        )
    pixels = np.median(np.stack(detections), axis=0)
    matrix = cv2.getPerspectiveTransform(
        np.float32(pixels), np.float32(table_corners(config))
    )
    return TableMapping(matrix)
