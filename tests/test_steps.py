"""Checking each step of a pick and place, so a failure says where it happened."""

import numpy as np
import pytest

from src.verification.steps import (
    StepCheck,
    first_failure,
    glass_at,
    glass_held,
    glass_lifted,
    glass_upright,
    hand_reached,
    others_still,
    step_summary,
    step_tally,
    tally_lines,
)

HAND = np.array([0.4, -0.1, 0.95])


def test_a_hand_on_its_target_has_reached() -> None:
    check = hand_reached("cup_a", HAND, HAND + [0.01, 0.0, 0.0])

    assert check == StepCheck("cup_a", "reach", True, check.detail)
    assert "1.0 cm" in check.detail


def test_a_hand_short_of_its_target_has_not_reached() -> None:
    assert not hand_reached("cup_a", HAND, HAND + [0.0, 0.05, 0.0]).passed


def test_a_glass_beside_the_hand_is_held() -> None:
    assert glass_held("cup_a", "carry", HAND + [0.0, 0.0, -0.03], HAND).passed


def test_a_glass_far_from_the_hand_was_dropped() -> None:
    check = glass_held("cup_a", "carry", HAND + [0.0, 0.0, -0.3], HAND)

    assert not check.passed
    assert check.step == "carry"


def test_a_glass_raised_off_the_table_is_lifted() -> None:
    assert glass_lifted("cup_a", 0.80, 0.90).passed
    assert not glass_lifted("cup_a", 0.80, 0.81).passed


def test_a_glass_is_at_its_target_measured_across_the_table_only() -> None:
    target = np.array([0.4, 0.0, 0.8])

    assert glass_at("cup_a", "place", target + [0.005, 0.0, 0.3], target).passed
    assert not glass_at("cup_a", "place", target + [0.03, 0.0, 0.0], target).passed


def test_a_glass_standing_straight_is_upright() -> None:
    assert glass_upright("cup_a", "release", np.array([0.0, 0.0, 1.0])).passed


def test_a_leaning_glass_is_not_upright() -> None:
    leaning = np.array([np.sin(np.radians(20)), 0.0, np.cos(np.radians(20))])

    check = glass_upright("cup_a", "release", leaning)

    assert not check.passed
    assert "20" in check.detail


def test_an_upside_down_glass_counts_as_upright_when_flipped() -> None:
    assert glass_upright("cup_d", "release", np.array([0.0, 0.0, -1.0]), True).passed
    assert not glass_upright("cup_d", "release", np.array([0.0, 0.0, 1.0]), True).passed


def test_untouched_glasses_are_still() -> None:
    before = {"cup_b": np.array([0.3, 0.1, 0.8])}

    assert others_still("cup_a", before, before).passed


def test_a_knocked_glass_is_named() -> None:
    before = {"cup_b": np.array([0.3, 0.1, 0.8]), "cup_c": np.array([0.5, 0.0, 0.8])}
    after = {**before, "cup_c": np.array([0.55, 0.0, 0.8])}

    check = others_still("cup_a", before, after)

    assert not check.passed
    assert "cup_c" in check.detail


def test_no_other_glasses_is_still() -> None:
    assert others_still("cup_a", {}, {}).passed


def test_the_first_failure_names_the_glass_and_step() -> None:
    checks = (
        StepCheck("cup_a", "grip", True, ""),
        StepCheck("cup_a", "carry", False, ""),
        StepCheck("cup_b", "grip", False, ""),
    )

    assert first_failure(checks) == "cup_a: carry"
    assert first_failure(checks[:1]) is None


def test_a_summary_is_plain_data_for_reports() -> None:
    summary = step_summary((StepCheck("cup_a", "grip", True, "held"),))

    assert summary == [
        {"glass": "cup_a", "step": "grip", "passed": True, "detail": "held"}
    ]


@pytest.mark.parametrize("step", ["carry", "twist"])
def test_held_checks_keep_their_step_name(step: str) -> None:
    assert glass_held("cup_d", step, HAND, HAND).step == step


def test_the_tally_counts_where_failed_runs_went_wrong() -> None:
    runs = [
        {"success": True, "failed_at": None},
        {"success": False, "failed_at": "cup_a: carry"},
        {"success": False, "failed_at": "cup_b: carry"},
        {"success": False, "failed_at": "cup_d: twist"},
        {"success": False, "failed_at": None},
    ]

    assert step_tally(runs) == {"carry": 2, "twist": 1, "final check": 1}


def test_tally_lines_name_each_step() -> None:
    lines = tally_lines([{"success": False, "failed_at": "cup_d: twist"}])

    assert lines == ["  first failed step: twist          1"]


def test_results_are_plain_booleans_even_from_numpy_numbers() -> None:
    check = glass_lifted("cup_a", np.float64(0.8), np.float64(0.9))

    assert type(check.passed) is bool
