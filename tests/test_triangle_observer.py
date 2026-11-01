"""Recognising the cup triangle in a video, and rebuilding it with the robot."""

from pathlib import Path

import numpy as np
import pytest

from src.perception.color_tracker import ColorSpec
from src.perception.homography import MarkerConfig
from src.perception.observe_video import TrackingConfig, observe_video
from src.pipeline.synthetic_demo import CameraView, render_synthetic_demo
from src.reasoning.demo_observer import ItemInfo, ObserverConfig
from src.reasoning.triangle_observer import is_stacked_between, observe_triangle
from src.schemas.plan import Plan, Relation, Step
from src.simulation.cup_task import build_triangle
from src.simulation.robot_env import RobotEnv, load_motion_config
from src.simulation.scene import ROOT, build_model, load_scene_config
from src.verification.triangle import load_triangle_config, triangle_targets

CONFIG = ROOT / "configs/cups.yaml"
FPS = 10.0
OBSERVER = ObserverConfig(move_threshold=0.05, stable_seconds=1.0)
CUPS = tuple(ItemInfo(n, (0.069, 0.069, 0.09), False) for n in ("a", "b", "c"))


def test_a_cup_raised_between_two_others_counts_as_stacked() -> None:
    assert is_stacked_between((300.0, 260.0), (250.0, 320.0), (350.0, 320.0))


def test_a_cup_level_with_the_others_is_not_stacked() -> None:
    assert not is_stacked_between((300.0, 318.0), (250.0, 320.0), (350.0, 320.0))


def test_a_raised_cup_off_to_one_side_is_not_stacked() -> None:
    assert not is_stacked_between((500.0, 260.0), (250.0, 320.0), (350.0, 320.0))


def test_stack_check_works_when_the_camera_is_rotated() -> None:
    assert is_stacked_between((320.0, 250.0), (250.0, 300.0), (350.0, 340.0))


def _moved(start, end, at: int, frames: int = 100) -> np.ndarray:
    track = np.tile(start, (frames, 1)).astype(float)
    track[at:] = end
    return track


def _tracks() -> tuple[dict, dict]:
    table = {
        "a": _moved((0.33, -0.05), (0.47, -0.09), 20),
        "b": _moved((0.33, -0.17), (0.47, -0.17), 40),
        "c": _moved((0.33, -0.29), (0.40, -0.13), 60),
    }
    pixels = {
        "a": _moved((200, 300), (260, 320), 20),
        "b": _moved((300, 300), (360, 320), 40),
        "c": _moved((400, 300), (310, 260), 60),
    }
    return table, pixels


def test_observer_finds_order_starts_and_where_to_build() -> None:
    table, pixels = _tracks()

    demo = observe_triangle(table, pixels, CUPS, FPS, OBSERVER)

    assert demo.order == ("a", "b", "c")
    assert demo.start_positions["c"] == pytest.approx((0.33, -0.29))
    assert demo.center == pytest.approx((0.47, -0.13))


def test_observer_rejects_a_video_without_a_stack() -> None:
    table, pixels = _tracks()
    pixels["c"] = _moved((400, 300), (310, 320), 60)

    with pytest.raises(ValueError, match="triangle"):
        observe_triangle(table, pixels, CUPS, FPS, OBSERVER)


def test_observer_rejects_a_video_where_a_cup_never_moved() -> None:
    table, pixels = _tracks()
    table["a"] = np.tile((0.33, -0.05), (100, 1)).astype(float)

    with pytest.raises(ValueError, match="a"):
        observe_triangle(table, pixels, CUPS, FPS, OBSERVER)


@pytest.mark.integration
def test_rendered_triangle_video_drives_the_robot_to_build_it(tmp_path: Path) -> None:
    config = load_scene_config(CONFIG)
    triangle = load_triangle_config(CONFIG)
    marker = MarkerConfig(center=(0.64, -0.10), side=0.10, marker_id=0)
    specs = [
        ColorSpec("cup_a", (22, 35), 100, 60),
        ColorSpec("cup_b", (45, 80), 100, 60),
        ColorSpec("cup_c", (170, 8), 100, 60),
    ]
    starts = {"cup_a": (0.37, -0.03), "cup_b": (0.35, -0.22), "cup_c": (0.37, -0.34)}
    table = config.table.top_height
    left, right, top = triangle_targets(triangle, table)
    video = tmp_path / "cups.mp4"
    render_synthetic_demo(
        config,
        marker,
        Plan(
            steps=tuple(
                Step(object=name, reference="table", relation=Relation.IN)
                for name in ("cup_a", "cup_b", "cup_c")
            )
        ),
        starts,
        {"cup_a": left[:2], "cup_b": right[:2], "cup_c": top[:2]},
        video,
        view=CameraView(azimuth=180.0, elevation=-50.0),
        target_heights={"cup_c": triangle.cup_height},
    )
    items = tuple(ItemInfo(i.name, i.size, False) for i in config.objects)
    tracking = TrackingConfig(min_area=150, area_fraction=0.3, frame_stride=3)

    seen = observe_video(video, specs, items, marker, tracking)
    demo = observe_triangle(seen.tracks, seen.pixels, items, seen.fps, OBSERVER)

    assert demo.order == ("cup_a", "cup_b", "cup_c")
    assert demo.center == pytest.approx(triangle.center, abs=0.03)

    model = build_model(config, demo.start_positions)
    env = RobotEnv(model, load_motion_config(CONFIG))
    result = build_triangle(env, config, demo.triangle_for(triangle), demo.order)

    assert result.success, result
