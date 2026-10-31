"""Check a test video before recording the rest, and say what to fix in plain language."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from src.perception.color_tracker import ColorSpec, track_colors
from src.perception.homography import (
    MarkerConfig,
    MarkerNotFoundError,
    estimate_mapping,
)

SAMPLE_FRAMES = 40
MIN_AREA = 150
AREA_FRACTION = 0.5
GOOD_MARKER_RATE = 0.9
GOOD_COLOR_RATE = 0.6
LOW_COLOR_RATE = 0.3
MIN_SHORT_SIDE = 720
MIN_SHARPNESS = 30.0
MIN_SECONDS = 5.0
Position = tuple[float, float]


class Status(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    PROBLEM = "PROBLEM"


@dataclass(frozen=True)
class WorkArea:
    """The table area the robot's arm can work in."""

    x: tuple[float, float]
    y: tuple[float, float]

    def contains(self, position: Position) -> bool:
        return (
            self.x[0] <= position[0] <= self.x[1]
            and self.y[0] <= position[1] <= self.y[1]
        )


@dataclass(frozen=True)
class VideoFacts:
    width: int
    height: int
    fps: float
    seconds: float
    marker_rate: float
    sharpness: float
    color_rates: Mapping[str, float]
    start_positions: Mapping[str, Position]


@dataclass(frozen=True)
class Finding:
    topic: str
    status: Status
    message: str


def _sample_frames(video: Path) -> tuple[list[np.ndarray], cv2.VideoCapture]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for index in np.linspace(0, max(total - 1, 0), min(SAMPLE_FRAMES, max(total, 1))):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    return frames, capture


def _marker_rate(frames: list[np.ndarray], marker: MarkerConfig) -> float:
    found = 0
    for frame in frames:
        try:
            estimate_mapping([frame], marker)
        except MarkerNotFoundError:
            continue
        found += 1
    return found / len(frames) if frames else 0.0


def _sharpness(frames: list[np.ndarray]) -> float:
    """Blurry images have few sharp edges, which gives a low Laplacian variance."""
    values = [
        cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        for frame in frames
    ]
    return float(np.median(values)) if values else 0.0


def _start_positions(
    frames: list[np.ndarray],
    tracks: Mapping[str, np.ndarray],
    depths: Mapping[str, float],
    marker: MarkerConfig,
) -> dict[str, Position]:
    try:
        mapping = estimate_mapping(frames, marker)
    except MarkerNotFoundError:
        return {}
    starts = {}
    for name, track in tracks.items():
        seen = track[~np.isnan(track[:, 0])]
        if len(seen):
            pixel = tuple(float(value) for value in seen[0])
            starts[name] = mapping.base_position(pixel, depths[name])
    return starts


def gather_facts(
    video: Path,
    specs: list[ColorSpec],
    depths: Mapping[str, float],
    marker: MarkerConfig,
) -> VideoFacts:
    """Measure what can be known from a sample of the video's frames."""
    frames, capture = _sample_frames(video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    tracks = track_colors(frames, specs, MIN_AREA, AREA_FRACTION)
    return VideoFacts(
        width=width,
        height=height,
        fps=fps,
        seconds=total / fps if fps else 0.0,
        marker_rate=_marker_rate(frames, marker),
        sharpness=_sharpness(frames),
        color_rates={
            name: float(np.mean(~np.isnan(track[:, 0])))
            for name, track in tracks.items()
        },
        start_positions=_start_positions(frames, tracks, depths, marker),
    )


def _marker_finding(rate: float) -> Finding:
    if rate >= GOOD_MARKER_RATE:
        return Finding("Marker", Status.OK, f"seen in {rate:.0%} of sampled frames")
    if rate > 0:
        return Finding(
            "Marker",
            Status.WARNING,
            f"seen in only {rate:.0%} of frames. Check nothing covers it, and try "
            "raising the camera.",
        )
    return Finding(
        "Marker",
        Status.PROBLEM,
        "not found. Check it's flat, in frame and in focus, and try a higher camera angle.",
    )


def _color_finding(name: str, rate: float) -> Finding:
    topic = f"Colour: {name}"
    if rate >= GOOD_COLOR_RATE:
        return Finding(topic, Status.OK, f"found in {rate:.0%} of sampled frames")
    if rate >= LOW_COLOR_RATE:
        return Finding(
            topic,
            Status.WARNING,
            f"found in only {rate:.0%} of frames. Your hand may be covering it, or "
            "the colour settings need adjusting.",
        )
    return Finding(
        topic,
        Status.PROBLEM,
        f"found in only {rate:.0%} of frames. Update its colour with scripts/read_colors.py.",
    )


def check_facts(facts: VideoFacts, area: WorkArea) -> tuple[Finding, ...]:
    """Turn measurements into a list of findings, and what to do about each."""
    short_side = min(facts.width, facts.height)
    findings = [
        Finding(
            "Resolution",
            Status.OK if short_side >= MIN_SHORT_SIDE else Status.WARNING,
            f"{facts.width} x {facts.height}"
            + ("" if short_side >= MIN_SHORT_SIDE else ". Use at least 720p."),
        ),
        Finding(
            "Length",
            Status.OK if facts.seconds >= MIN_SECONDS else Status.WARNING,
            f"{facts.seconds:.0f} seconds",
        ),
        Finding(
            "Sharpness",
            Status.OK if facts.sharpness >= MIN_SHARPNESS else Status.WARNING,
            f"score {facts.sharpness:.0f}"
            + (
                ""
                if facts.sharpness >= MIN_SHARPNESS
                else ". Lock focus, and add light."
            ),
        ),
        _marker_finding(facts.marker_rate),
    ]
    findings += [_color_finding(n, r) for n, r in facts.color_rates.items()]
    for name, position in facts.start_positions.items():
        inside = area.contains(position)
        findings.append(
            Finding(
                f"Reach: {name}",
                Status.OK if inside else Status.PROBLEM,
                f"at ({position[0] * 100:.0f}, {position[1] * 100:.0f}) cm"
                + ("" if inside else ", outside the robot's reach. Move it closer."),
            )
        )
    return tuple(findings)
