"""End to end: a rendered demo video, with known positions, turned into a plan."""

from pathlib import Path

import numpy as np
import pytest

from src.perception.color_tracker import ColorSpec
from src.perception.homography import MarkerConfig
from src.perception.observe_video import (
    TrackingConfig,
    items_from_scene,
    observe_video,
)
from src.pipeline.synthetic_demo import render_synthetic_demo
from src.reasoning.demo_observer import ObserverConfig, observe_demo
from src.schemas.plan import Plan, Relation, Step
from src.simulation.scene import load_scene_config

MARKER = MarkerConfig(center=(0.64, -0.10), side=0.10, marker_id=0)
SPECS = [
    ColorSpec("apple", (170, 8), 100, 60),
    ColorSpec("lemon", (22, 35), 100, 60),
    ColorSpec("lime", (45, 80), 100, 60),
    ColorSpec("box", (9, 20), 100, 60),
]
STARTS = {
    "apple": (0.35, -0.28),
    "lemon": (0.33, -0.15),
    "lime": (0.36, -0.03),
    "box": (0.50, -0.21),
    "bowl": (0.50, 0.03),
}
TRUTH = Plan(
    steps=(
        Step(object="lemon", reference="box", relation=Relation.IN),
        Step(object="apple", reference="box", relation=Relation.IN),
    )
)
TARGETS = {"apple": (0.50, -0.26), "lemon": (0.50, -0.16)}
TRACKING = TrackingConfig(min_area=150, area_fraction=0.5, frame_stride=3)
OBSERVER = ObserverConfig(move_threshold=0.05, stable_seconds=1.0)


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    config = load_scene_config()
    output = tmp_path_factory.mktemp("demo") / "demo.mp4"
    render_synthetic_demo(config, MARKER, TRUTH, STARTS, TARGETS, output)
    return output


@pytest.mark.integration
def test_video_gives_the_correct_plan_and_positions(video: Path) -> None:
    config = load_scene_config()
    # The white tray has no colour to follow, so it's given directly.
    items = tuple(i for i in items_from_scene(config) if i.name != "bowl")

    seen = observe_video(video, SPECS, items, MARKER, TRACKING)
    demo = observe_demo(seen.tracks, items, seen.fps, OBSERVER)

    assert demo.plan == TRUTH
    for name in ("apple", "lemon", "lime", "box"):
        error = np.linalg.norm(
            np.array(demo.start_positions[name]) - np.array(STARTS[name])
        )
        assert error < 0.03, f"{name} start is off by {error * 100:.1f} cm"
