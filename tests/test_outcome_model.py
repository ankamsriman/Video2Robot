"""The outcome model learns from runs which way of doing a layout tends to work."""

import numpy as np
import pytest

from src.reasoning.outcome_model import features, table_cm, train

STARTS = [(18.0, 10.0), (30.0, -12.0), (22.0, 20.0)]


def _runs(rng: np.random.Generator, count: int):
    """Made-up runs: two arms work near the middle, one arm works far to the side."""
    rows = []
    for _ in range(count):
        tower = (rng.uniform(24, 36), rng.uniform(-24, 24))
        near_middle = abs(tower[1]) < 12
        rows.append((features(STARTS, tower, "two"), near_middle))
        rows.append((features(STARTS, tower, "nearest"), not near_middle))
    return rows


def test_features_have_the_same_length_for_both_modes() -> None:
    assert features(STARTS, (30, 5), "two").shape == features(STARTS, (30, 5), "nearest").shape


def test_the_model_learns_which_mode_suits_which_layout() -> None:
    model = train(_runs(np.random.default_rng(0), 200))

    assert model.best_mode(STARTS, (30.0, 2.0)) == "two"
    assert model.best_mode(STARTS, (30.0, 22.0)) == "nearest"


def test_robot_positions_become_table_centimetres() -> None:
    assert table_cm((0.46, -0.12), 0.16) == pytest.approx((30.0, -12.0))


def test_chances_are_between_zero_and_one() -> None:
    model = train(_runs(np.random.default_rng(1), 50))
    chance = model.chance(STARTS, (30.0, 10.0), "two")

    assert 0.0 <= chance <= 1.0


def test_two_arms_win_a_near_tie() -> None:
    from src.reasoning.outcome_model import prefer_two_arms

    assert prefer_two_arms({"nearest": 0.48, "two": 0.46}, margin=0.10)


def test_one_arm_wins_only_when_clearly_better() -> None:
    from src.reasoning.outcome_model import prefer_two_arms

    assert not prefer_two_arms({"nearest": 0.80, "two": 0.60}, margin=0.10)
    assert prefer_two_arms({"nearest": 0.80, "two": 0.60}, margin=0.25)


def test_no_margin_keeps_the_plain_comparison() -> None:
    from src.reasoning.outcome_model import prefer_two_arms

    assert not prefer_two_arms({"nearest": 0.48, "two": 0.46}, margin=0.0)
