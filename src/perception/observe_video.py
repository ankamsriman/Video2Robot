"""Turn a demo video into per-frame table positions of every item."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.perception.color_tracker import ColorSpec, track_colors
from src.perception.homography import MarkerConfig, TableMapping, estimate_mapping
from src.reasoning.demo_observer import ItemInfo
from src.simulation.scene import CONFIG_PATH, SceneConfig

MARKER_SEARCH_FRAMES = 30
MARKER_SEARCH_STRIDE = 3


@dataclass(frozen=True)
class TrackingConfig:
    min_area: int
    area_fraction: float
    frame_stride: int


@dataclass(frozen=True)
class VideoObservation:
    tracks: Mapping[str, np.ndarray]
    pixels: Mapping[str, np.ndarray]
    fps: float
    mapping: TableMapping


def load_marker_config(path: Path = CONFIG_PATH) -> MarkerConfig:
    raw = yaml.safe_load(path.read_text())["marker"]
    return MarkerConfig(tuple(raw["center"]), raw["side"], raw["marker_id"])


def load_tracking_config(path: Path = CONFIG_PATH) -> TrackingConfig:
    return TrackingConfig(**yaml.safe_load(path.read_text())["tracking"])


def load_color_specs(path: Path = CONFIG_PATH) -> list[ColorSpec]:
    colors = yaml.safe_load(path.read_text())["colors"]
    return [
        ColorSpec(name, tuple(spec["hue"]), spec["min_saturation"], spec["min_value"])
        for name, spec in colors.items()
    ]


def items_from_scene(config: SceneConfig) -> tuple[ItemInfo, ...]:
    objects = tuple(ItemInfo(i.name, i.size, False) for i in config.objects)
    containers = tuple(ItemInfo(i.name, i.size, True) for i in config.containers)
    return objects + containers


def _frames(video: Path, stride: int) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            if index % stride == 0:
                yield frame
            index += 1
    finally:
        capture.release()


def _video_fps(video: Path) -> float:
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return fps


def _marker_frames(video: Path) -> list[np.ndarray]:
    frames = []
    for frame in _frames(video, MARKER_SEARCH_STRIDE):
        frames.append(frame)
        if len(frames) >= MARKER_SEARCH_FRAMES:
            break
    return frames


def _to_table(pixels: np.ndarray, mapping: TableMapping, depth: float) -> np.ndarray:
    positions = np.full_like(pixels, np.nan)
    for index, pixel in enumerate(pixels):
        if not np.isnan(pixel[0]):
            positions[index] = mapping.base_position(tuple(pixel), depth)
    return positions


def observe_video(
    video: Path,
    specs: list[ColorSpec],
    items: tuple[ItemInfo, ...],
    marker: MarkerConfig,
    tracking: TrackingConfig,
) -> VideoObservation:
    """Find the marker, follow each coloured item, and convert to table positions."""
    mapping = estimate_mapping(_marker_frames(video), marker)
    pixels = track_colors(
        _frames(video, tracking.frame_stride),
        specs,
        tracking.min_area,
        tracking.area_fraction,
    )
    depths = {item.name: item.size[0] for item in items}
    tracks = {
        name: _to_table(track, mapping, depths[name]) for name, track in pixels.items()
    }
    return VideoObservation(
        tracks, pixels, _video_fps(video) / tracking.frame_stride, mapping
    )
