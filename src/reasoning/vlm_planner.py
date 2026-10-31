"""Write the plan by asking a vision AI model small questions about key frames.

This follows the question sequence published in the SeeDo paper. Any model can be used,
as long as it has an `ask` method, so the choice of provider stays open.
"""

from collections.abc import Callable, Sequence
from typing import Protocol

from src.schemas.plan import Plan, Relation, Step

MANIPULATING = "hand is manipulating"
NOT_MANIPULATING = "hand is not manipulating"
RELATION_PHRASES = (
    ("on top of", Relation.ON_TOP_OF),
    ("at the back of", Relation.AT_THE_BACK_OF),
    ("in front of", Relation.IN_FRONT_OF),
    ("to the left", Relation.TO_THE_LEFT),
    ("to the right", Relation.TO_THE_RIGHT),
    (" in ", Relation.IN),
)

FILTER_SYSTEM = (
    "You are an operations inspector checking whether a hand is holding an object."
)
FILTER_USER = (
    "This is a frame from a pick-and-place task. Respond with exactly "
    "'Hand is manipulating an object' or 'Hand is not manipulating an object'."
)
PICK_SYSTEM = "You are an inspector identifying which object is being picked up."
PICK_USER = (
    "This frame shows the pick moment. The items are: {items}. Their outlines and labels "
    "are only a guide. Which object is being picked? Respond as 'Object Picked: <name>'."
)
REFERENCE_SYSTEM = (
    "You are an inspector finding the reference for a placement. In a storage task, it's "
    "the container the object is placed into."
)
REFERENCE_USER = (
    "This frame shows the placing moment. The items are: {items}. {picked} is being "
    "placed. Which other item is the reference? Respond as 'Reference Object: <name>'."
)
RELATION_SYSTEM = "You describe spatial relations in images, accurately."
RELATION_USER = (
    "{picked} is being placed near {reference}. Directions mean the visual direction of "
    "the picture. Choose one: in, on top of, at the back of, in front of, to the left, "
    "to the right. Respond as 'Drop {picked} <relation> {reference}'."
)
MeasuredRelation = Callable[[str, str], Relation]


class VisionModel(Protocol):
    def ask(self, system: str, user: str, image: bytes) -> str: ...


class PlanningError(ValueError):
    """The AI's answer couldn't be turned into a valid plan."""


def match_name(answer: str, items: Sequence[str]) -> str:
    """Find the one scene item named in an answer, rejecting anything ambiguous."""
    text = answer.lower()
    found = [item for item in items if item.lower() in text]
    if len(found) != 1:
        raise PlanningError(f"Expected one of {list(items)} in the answer: {answer!r}")
    return found[0]


def parse_relation(answer: str) -> Relation:
    text = f" {answer.lower().strip(' .')} "
    for phrase, relation in RELATION_PHRASES:
        if phrase in text:
            return relation
    raise PlanningError(f"No known relation in the answer: {answer!r}")


def _working_frames(model: VisionModel, frames: Sequence[bytes]) -> list[bytes]:
    kept = []
    for frame in frames:
        answer = model.ask(FILTER_SYSTEM, FILTER_USER, frame).lower()
        if MANIPULATING in answer and NOT_MANIPULATING not in answer:
            kept.append(frame)
    return kept


def _plan_step(
    model: VisionModel,
    pick_frame: bytes,
    place_frame: bytes,
    items: Sequence[str],
    measured_relation: MeasuredRelation | None,
) -> Step:
    listed = ", ".join(items)
    picked = match_name(
        model.ask(PICK_SYSTEM, PICK_USER.format(items=listed), pick_frame), items
    )
    others = [item for item in items if item != picked]
    reference = match_name(
        model.ask(
            REFERENCE_SYSTEM,
            REFERENCE_USER.format(items=listed, picked=picked),
            place_frame,
        ),
        others,
    )
    answer = model.ask(
        RELATION_SYSTEM,
        RELATION_USER.format(picked=picked, reference=reference),
        place_frame,
    )
    relation = (
        measured_relation(picked, reference)
        if measured_relation
        else parse_relation(answer)
    )
    return Step(object=picked, reference=reference, relation=relation)


def plan_with_model(
    model: VisionModel,
    frames: Sequence[bytes],
    items: Sequence[str],
    measured_relation: MeasuredRelation | None = None,
) -> Plan:
    """Filter key frames, pair them as pick then place, and ask about each pair.

    If `measured_relation` is given, measured positions decide the relation instead of
    the AI. The AI's answer is still requested, so that both can be compared.
    """
    working = _working_frames(model, frames)
    if len(working) % 2:
        raise PlanningError(
            f"{len(working)} working frames can't pair into pick and place steps"
        )
    steps = tuple(
        _plan_step(model, pick, place, items, measured_relation)
        for pick, place in zip(working[::2], working[1::2], strict=True)
    )
    if not steps:
        raise PlanningError("No frames showed a hand working on an object")
    return Plan(steps=steps)
