"""The plan format shared by the hand-written plans and, later, the AI planner."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.schemas.plan import Plan, Relation, Step, load_plan, save_plan


def test_step_accepts_all_six_seedo_relations() -> None:
    assert {relation.value for relation in Relation} == {
        "in",
        "on_top_of",
        "at_the_back_of",
        "in_front_of",
        "to_the_left",
        "to_the_right",
    }


def test_step_rejects_unknown_relation() -> None:
    with pytest.raises(ValidationError):
        Step(object="apple", reference="bowl", relation="beside")


def test_step_rejects_object_placed_relative_to_itself() -> None:
    with pytest.raises(ValidationError):
        Step(object="apple", reference="apple", relation=Relation.IN)


def test_plan_requires_at_least_one_step() -> None:
    with pytest.raises(ValidationError):
        Plan(steps=[])


def test_plan_cannot_be_modified_after_creation() -> None:
    plan = Plan(steps=[Step(object="apple", reference="bowl", relation=Relation.IN)])

    with pytest.raises(ValidationError):
        plan.steps = ()


def test_plan_survives_saving_and_loading(tmp_path: Path) -> None:
    plan = Plan(steps=[Step(object="apple", reference="bowl", relation=Relation.IN)])
    path = tmp_path / "plan.json"

    save_plan(plan, path)

    assert load_plan(path) == plan
