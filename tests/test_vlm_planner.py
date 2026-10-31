"""The AI planner's logic, tested with a scripted fake so no API key is needed."""

import pytest

from src.reasoning.vlm_planner import (
    PlanningError,
    match_name,
    parse_relation,
    plan_with_model,
)
from src.schemas.plan import Plan, Relation, Step

ITEMS = ("apple", "lemon", "box", "bowl")
FRAMES = [b"f0", b"f1", b"f2", b"f3", b"f4"]


class FakeModel:
    """Answers each question from a queue, and records what it was asked."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.questions: list[tuple[str, bytes]] = []

    def ask(self, system: str, user: str, image: bytes) -> str:
        self.questions.append((user, image))
        return self.answers.pop(0)


USED = "Hand is manipulating an object"
UNUSED = "Hand is not manipulating an object"


def test_match_name_finds_items_in_a_wordy_answer() -> None:
    assert match_name("Object Picked: the red Apple.", ITEMS) == "apple"


def test_match_name_rejects_something_not_in_the_scene() -> None:
    with pytest.raises(PlanningError, match="banana"):
        match_name("a banana", ITEMS)


def test_match_name_rejects_an_answer_naming_two_items() -> None:
    with pytest.raises(PlanningError):
        match_name("the apple or the lemon", ITEMS)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Drop apple in the box", Relation.IN),
        ("on top of the box", Relation.ON_TOP_OF),
        ("at the back of the bowl", Relation.AT_THE_BACK_OF),
        ("in front of the bowl", Relation.IN_FRONT_OF),
        ("to the left of the lemon", Relation.TO_THE_LEFT),
        ("to the right of the lemon", Relation.TO_THE_RIGHT),
    ],
)
def test_parse_relation_reads_the_six_relations(
    answer: str, expected: Relation
) -> None:
    assert parse_relation(answer) is expected


def test_parse_relation_rejects_an_unknown_answer() -> None:
    with pytest.raises(PlanningError):
        parse_relation("somewhere nearby")


def test_frames_without_a_working_hand_are_filtered_out() -> None:
    model = FakeModel(
        [
            UNUSED,
            USED,
            USED,
            UNUSED,
            UNUSED,
            "Object Picked: apple",
            "Reference Object: box",
            "Drop apple in the box",
        ]
    )

    plan = plan_with_model(model, FRAMES, ITEMS)

    assert plan == Plan(
        steps=(Step(object="apple", reference="box", relation=Relation.IN),)
    )
    assert [image for _, image in model.questions[5:]] == [b"f1", b"f2", b"f2"]


def test_measured_relation_overrides_the_ai_when_provided() -> None:
    model = FakeModel(
        [USED, USED, "apple", "box", "to the left"],
    )

    plan = plan_with_model(
        model, FRAMES[:2], ITEMS, measured_relation=lambda obj, ref: Relation.IN
    )

    assert plan.steps[0].relation is Relation.IN


def test_an_odd_number_of_working_frames_is_reported() -> None:
    model = FakeModel([USED])

    with pytest.raises(PlanningError, match="pair"):
        plan_with_model(model, FRAMES[:1], ITEMS)
