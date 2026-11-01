"""The video chain must survive the damage a phone adds, with colours picked per video."""

from collections.abc import Callable
from pathlib import Path

import cv2
import imageio
import numpy as np
import pytest

from src.perception.color_tracker import ColorSpec, _largest_blob, _mask
from src.perception.observe_video import (
    items_from_scene,
    load_marker_config,
    observe_video,
)
from src.perception.read_colors import suggest_spec
from src.pipeline.synthetic_demo import render_synthetic_demo
from src.reasoning.demo_observer import observe_demo
from src.simulation.scene import load_scene_config
from tests.test_video_to_plan import OBSERVER, SPECS, STARTS, TARGETS, TRACKING, TRUTH

MAX_BLOCK_ERROR = 0.02
PATCH_RADIUS = 4
BRIGHT_ENOUGH = 150
Damage = Callable[[np.ndarray, int], np.ndarray]


def _dim(frame: np.ndarray, _: int) -> np.ndarray:
    return (frame * 0.6).astype(np.uint8)


def _noise(frame: np.ndarray, index: int) -> np.ndarray:
    grain = np.random.default_rng(index).normal(0, 12, frame.shape)
    return np.clip(frame + grain, 0, 255).astype(np.uint8)


def _compressed(frame: np.ndarray, _: int) -> np.ndarray:
    encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 25])[1]
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _cool_light(frame: np.ndarray, _: int) -> np.ndarray:
    return np.clip(frame * np.array([0.8, 1.0, 1.15]), 0, 255).astype(np.uint8)


def _pick_colors(damaged: np.ndarray, clean: np.ndarray) -> list[ColorSpec]:
    """Sample each object's lit face, as scripts/read_colors.py does on a real video."""
    hsv = cv2.cvtColor(cv2.cvtColor(clean, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    picture = cv2.cvtColor(damaged, cv2.COLOR_RGB2BGR)
    specs = []
    for spec in SPECS:
        lit = ColorSpec(spec.name, spec.hue, spec.min_saturation, BRIGHT_ENOUGH)
        blob, _ = _largest_blob(_mask(hsv, lit))
        rows, columns = np.nonzero(blob)
        row, column = int(rows.mean()), int(columns.mean())
        patch = picture[
            row - PATCH_RADIUS : row + PATCH_RADIUS + 1,
            column - PATCH_RADIUS : column + PATCH_RADIUS + 1,
        ]
        specs.append(suggest_spec(spec.name, patch).spec)
    return specs


@pytest.fixture(scope="module")
def clean_frames(tmp_path_factory: pytest.TempPathFactory) -> list[np.ndarray]:
    video = tmp_path_factory.mktemp("clean") / "clean.mp4"
    render_synthetic_demo(
        load_scene_config(), load_marker_config(), TRUTH, STARTS, TARGETS, video
    )
    return imageio.mimread(video, memtest=False)


@pytest.mark.integration
@pytest.mark.parametrize("damage", [_dim, _noise, _compressed, _cool_light])
def test_plan_and_positions_survive_damage(
    damage: Damage, clean_frames: list[np.ndarray], tmp_path: Path
) -> None:
    damaged = [damage(frame, index) for index, frame in enumerate(clean_frames)]
    video = tmp_path / "damaged.mp4"
    with imageio.get_writer(video, fps=30, macro_block_size=1) as writer:
        for frame in damaged:
            writer.append_data(frame)
    config = load_scene_config()
    items = tuple(i for i in items_from_scene(config) if i.name != "bowl")
    specs = _pick_colors(damaged[0], clean_frames[0])

    seen = observe_video(video, specs, items, load_marker_config(), TRACKING)
    demo = observe_demo(seen.tracks, items, seen.fps, OBSERVER)

    assert demo.plan == TRUTH
    for name in ("apple", "lemon", "lime"):
        error = np.linalg.norm(
            np.array(demo.start_positions[name]) - np.array(STARTS[name])
        )
        assert error < MAX_BLOCK_ERROR, f"{name} is off by {error * 100:.1f} cm"
