"""Working out the plan from where objects moved in the video, with no AI model."""

import numpy as np
import pytest

from src.reasoning.demo_observer import ItemInfo, ObserverConfig, observe_demo
from src.schemas.plan import Relation, Step

FPS = 10.0
CONFIG = ObserverConfig(move_threshold=0.05, stable_seconds=1.0)
ITEMS = (
    ItemInfo("apple", (0.06, 0.07, 0.07), False),
    ItemInfo("lemon", (0.06, 0.07, 0.07), False),
    ItemInfo("box", (0.15, 0.18, 0.04), True),
    ItemInfo("bowl", (0.15, 0.18, 0.04), True),
)
BOX = (0.50, -0.21)
BOWL = (0.50, 0.03)
FRAMES = 100


def _still(position: tuple[float, float]) -> np.ndarray:
    return np.tile(position, (FRAMES, 1)).astype(float)


def _moved(
    start: tuple[float, float], end: tuple[float, float], at: int, hidden: int = 10
) -> np.ndarray:
    """An object that's carried at frame `at`, hidden by the hand while it moves."""
    track = _still(start)
    track[at : at + hidden] = np.nan
    track[at + hidden :] = end
    return track


def _tracks(**moves: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "apple": _still((0.35, -0.28)),
        "lemon": _still((0.35, -0.16)),
        "box": _still(BOX),
        "bowl": _still(BOWL),
        **moves,
    }


def test_an_object_moved_into_a_container_becomes_an_in_step() -> None:
    tracks = _tracks(apple=_moved((0.35, -0.28), (0.50, -0.23), at=30))

    demo = observe_demo(tracks, ITEMS, FPS, CONFIG)

    assert demo.plan.steps == (Step(object="apple", reference="box", relation="in"),)


def test_steps_are_ordered_by_when_each_object_moved() -> None:
    tracks = _tracks(
        apple=_moved((0.35, -0.28), (0.50, -0.25), at=60),
        lemon=_moved((0.35, -0.16), (0.50, 0.01), at=20),
    )

    demo = observe_demo(tracks, ITEMS, FPS, CONFIG)

    assert [step.object for step in demo.plan.steps] == ["lemon", "apple"]


def test_starting_positions_come_from_the_video() -> None:
    tracks = _tracks(apple=_moved((0.37, -0.26), (0.50, -0.23), at=30))

    demo = observe_demo(tracks, ITEMS, FPS, CONFIG)

    assert demo.start_positions["apple"] == pytest.approx((0.37, -0.26))
    assert demo.start_positions["box"] == pytest.approx(BOX)


def test_small_tracking_jitter_is_not_a_move() -> None:
    jitter = _still((0.35, -0.16)) + 0.005 * np.sin(np.arange(FRAMES))[:, None]
    tracks = _tracks(
        apple=_moved((0.35, -0.28), (0.50, -0.23), at=30),
        lemon=jitter,
    )

    demo = observe_demo(tracks, ITEMS, FPS, CONFIG)

    assert [step.object for step in demo.plan.steps] == ["apple"]


def test_an_object_placed_beside_another_uses_a_direction() -> None:
    tracks = _tracks(apple=_moved((0.35, -0.28), (0.35, -0.04), at=30))

    demo = observe_demo(tracks, ITEMS, FPS, CONFIG)

    step = demo.plan.steps[0]
    assert (step.reference, step.relation) == ("lemon", Relation.TO_THE_LEFT)


def test_a_video_where_nothing_moved_gives_a_clear_error() -> None:
    with pytest.raises(ValueError, match="moved"):
        observe_demo(_tracks(), ITEMS, FPS, CONFIG)


def test_an_item_never_seen_at_the_start_gives_a_clear_error() -> None:
    tracks = _tracks(apple=np.full((FRAMES, 2), np.nan))

    with pytest.raises(ValueError, match="apple"):
        observe_demo(tracks, ITEMS, FPS, CONFIG)
