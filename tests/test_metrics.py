"""Scoring a predicted plan against the correct plan."""

import pytest

from src.evaluation.metrics import ErrorType, classify_error, fsr, ssr, tsr
from src.schemas.plan import Plan, Relation, Step


def _step(obj: str, ref: str = "box", relation: Relation = Relation.IN) -> Step:
    return Step(object=obj, reference=ref, relation=relation)


def _plan(*steps: Step) -> Plan:
    return Plan(steps=steps)


A, B, C = _step("a"), _step("b"), _step("c", "bowl")


def test_tsr_needs_the_exact_steps_in_order() -> None:
    assert tsr(_plan(A, B, C), _plan(A, B, C)) == 1
    assert tsr(_plan(A, C, B), _plan(A, B, C)) == 0


def test_fsr_ignores_order_but_not_the_final_arrangement() -> None:
    assert fsr(_plan(A, C, B), _plan(A, B, C)) == 1
    assert fsr(_plan(A, B), _plan(A, B, C)) == 0


def test_fsr_uses_the_last_move_of_an_object() -> None:
    moved_twice = _plan(_step("a", "bowl"), A)

    assert fsr(moved_twice, _plan(A)) == 1


def test_ssr_counts_steps_matched_in_order() -> None:
    assert ssr(_plan(A, C, B), _plan(A, B, C)) == pytest.approx(2 / 3)


def test_ssr_never_goes_backward_after_a_later_match() -> None:
    assert ssr(_plan(C, B), _plan(A, B, C)) == pytest.approx(1 / 3)


def test_ssr_ignores_extra_predicted_steps() -> None:
    assert ssr(_plan(A, _step("x"), B, C), _plan(A, B, C)) == 1.0


def test_correct_plan_has_no_error() -> None:
    assert classify_error(_plan(A, B), _plan(A, B)) is None


def test_wrong_object_is_a_vision_error() -> None:
    assert classify_error(_plan(_step("x")), _plan(A)) is ErrorType.VISION


def test_wrong_relation_or_reference_is_a_spatial_error() -> None:
    wrong = _step("a", "box", Relation.TO_THE_LEFT)

    assert classify_error(_plan(wrong), _plan(A)) is ErrorType.SPATIAL
    assert classify_error(_plan(_step("a", "bowl")), _plan(A)) is ErrorType.SPATIAL


def test_wrong_order_or_count_is_a_temporal_error() -> None:
    assert classify_error(_plan(B, A), _plan(A, B)) is ErrorType.TEMPORAL
    assert classify_error(_plan(A), _plan(A, B)) is ErrorType.TEMPORAL


def test_ssr_finds_the_longest_in_order_match_not_the_first_greedy_one() -> None:
    assert ssr(_plan(C, A, B), _plan(A, B, C)) == pytest.approx(2 / 3)
