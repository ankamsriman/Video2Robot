"""The robot plan format: an ordered list of 'move this object relative to that one' steps."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Relation(str, Enum):
    """The six spatial relations defined by the SeeDo paper."""

    IN = "in"
    ON_TOP_OF = "on_top_of"
    AT_THE_BACK_OF = "at_the_back_of"
    IN_FRONT_OF = "in_front_of"
    TO_THE_LEFT = "to_the_left"
    TO_THE_RIGHT = "to_the_right"


class Step(BaseModel):
    """One pick-and-place: pick up `object`, put it `relation` to `reference`."""

    model_config = ConfigDict(frozen=True)

    object: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    relation: Relation

    @model_validator(mode="after")
    def _object_differs_from_reference(self) -> "Step":
        if self.object == self.reference:
            raise ValueError(f"'{self.object}' cannot be placed relative to itself")
        return self


class Plan(BaseModel):
    """An ordered, unchangeable sequence of steps."""

    model_config = ConfigDict(frozen=True)

    steps: tuple[Step, ...] = Field(min_length=1)


def save_plan(plan: Plan, path: Path) -> None:
    path.write_text(plan.model_dump_json(indent=2))


def load_plan(path: Path) -> Plan:
    return Plan.model_validate_json(path.read_text())
