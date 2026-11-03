"""A typed instruction, turned into a plan the code checks before the robot acts.

A vision language model reads the instruction and a picture, and must answer with one of a
few allowed tasks and orders. The positions still come from measuring the video, because
such models are rarely accurate to the centimetre.
"""

import json
import re
from collections.abc import Sequence
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

GLASSES_IN_TRIANGLE = 3


class InstructionError(ValueError):
    """The instruction couldn't be turned into a task the robot can do."""


class Task(str, Enum):
    TRIANGLE = "triangle"
    FLIPPED_TOWER = "flipped_tower"


class Order(str, Enum):
    """Which glasses go first. The first two become the lower glasses, the third the top."""

    AS_IN_VIDEO = "as_in_video"
    NEAREST_FIRST = "nearest_first"
    FARTHEST_FIRST = "farthest_first"
    LEFT_FIRST = "left_first"
    RIGHT_FIRST = "right_first"


class InstructionPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: Task
    order: Order
    understood: bool
    reason: str = Field(default="", max_length=400)


def parse_plan(reply: str) -> InstructionPlan:
    """Read the model's answer, refusing anything outside the allowed tasks and orders."""
    found = re.search(r"\{.*\}", reply, re.DOTALL)
    if found is None:
        raise InstructionError("The model's answer had no plan in it")
    try:
        return InstructionPlan.model_validate(json.loads(found.group(0)))
    except (json.JSONDecodeError, ValidationError) as error:
        raise InstructionError(f"The model's plan wasn't one the robot can do: {error}") from error


def _sort_key(order: Order):
    """x is forward from the robot, y is to its left, which is also the person's left."""
    return {
        Order.NEAREST_FIRST: lambda start: start[0],
        Order.FARTHEST_FIRST: lambda start: -start[0],
        Order.LEFT_FIRST: lambda start: -start[1],
        Order.RIGHT_FIRST: lambda start: start[1],
    }.get(order)


def choose_glasses(
    starts: Sequence[np.ndarray], plan: InstructionPlan, video_has_flip: bool
) -> tuple[tuple[int, ...], bool]:
    """Which glasses to use, in which order, and whether the last one is flipped.

    In a video with a flip, the fourth glass is the one shown flipped. A triangle uses only
    the other three. For a flipped tower, the order covers all four glasses, and the last
    one is flipped, so "nearest first" flips the farthest glass. Keeping the video's order
    flips the glass the person flipped.
    """
    if not plan.understood:
        raise InstructionError(f"The model didn't understand the instruction: {plan.reason}")
    wants_flip = plan.task is Task.FLIPPED_TOWER
    if wants_flip and not video_has_flip:
        raise InstructionError(
            "A flipped tower needs a video that shows a fourth glass being flipped"
        )
    used = list(range(GLASSES_IN_TRIANGLE + 1 if wants_flip else GLASSES_IN_TRIANGLE))
    key = _sort_key(plan.order)
    if key is not None:
        used.sort(key=lambda index: float(key(starts[index])))
    return tuple(used), wants_flip
