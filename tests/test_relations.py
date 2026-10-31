"""Geometry shared by placing objects and by checking results afterwards."""

import pytest

from src.schemas.plan import Relation
from src.verification.relations import (
    Box,
    classify_relation,
    free_slot,
    placement_target,
)

BOWL = Box(center=(0.30, 0.00, 0.75), half_size=(0.07, 0.07, 0.02))
BLOCK = Box(center=(0.30, 0.20, 0.75), half_size=(0.025, 0.025, 0.03))
GAP = 0.02


def test_in_targets_the_reference_center() -> None:
    assert placement_target(BLOCK, BOWL, Relation.IN, GAP)[:2] == pytest.approx(
        (0.30, 0.00)
    )


def test_on_top_of_rests_on_the_reference() -> None:
    x, y, z = placement_target(BLOCK, BOWL, Relation.ON_TOP_OF, GAP)

    assert (x, y) == pytest.approx((0.30, 0.00))
    assert z == pytest.approx(0.75 + 0.02 + 0.03)


@pytest.mark.parametrize(
    ("relation", "dx", "dy"),
    [
        (Relation.AT_THE_BACK_OF, 0.115, 0.0),
        (Relation.IN_FRONT_OF, -0.115, 0.0),
        (Relation.TO_THE_LEFT, 0.0, 0.115),
        (Relation.TO_THE_RIGHT, 0.0, -0.115),
    ],
)
def test_side_relations_leave_a_gap_between_the_objects(
    relation: Relation, dx: float, dy: float
) -> None:
    x, y, _ = placement_target(BLOCK, BOWL, relation, GAP)

    assert (x, y) == pytest.approx((0.30 + dx, 0.00 + dy))


def test_object_inside_footprint_at_low_height_is_in() -> None:
    placed = Box(center=(0.31, 0.01, 0.76), half_size=(0.025, 0.025, 0.03))

    assert classify_relation(placed, BOWL) is Relation.IN


def test_object_resting_above_a_flat_reference_is_on_top_of() -> None:
    base = Box(center=(0.30, 0.00, 0.75), half_size=(0.03, 0.03, 0.03))
    placed = Box(center=(0.30, 0.00, 0.81), half_size=(0.03, 0.03, 0.03))

    assert classify_relation(placed, base, container=False) is Relation.ON_TOP_OF


@pytest.mark.parametrize(
    ("dx", "dy", "expected"),
    [
        (0.2, 0.01, Relation.AT_THE_BACK_OF),
        (-0.2, 0.01, Relation.IN_FRONT_OF),
        (0.01, 0.2, Relation.TO_THE_LEFT),
        (0.01, -0.2, Relation.TO_THE_RIGHT),
    ],
)
def test_outside_objects_use_their_dominant_direction(
    dx: float, dy: float, expected: Relation
) -> None:
    placed = Box(center=(0.30 + dx, dy, 0.75), half_size=(0.025, 0.025, 0.03))

    assert classify_relation(placed, BOWL) is expected


def test_boxes_are_immutable() -> None:
    with pytest.raises(AttributeError):
        BOWL.center = (0.0, 0.0, 0.0)


def test_free_slot_in_an_empty_container_stays_inside_the_walls() -> None:
    x, y = free_slot(BLOCK, BOWL, occupants=())

    assert x == pytest.approx(0.30)
    assert abs(y) + BLOCK.half_size[1] <= BOWL.half_size[1]


def test_free_slot_does_not_overlap_an_object_already_in_the_container() -> None:
    first_x, first_y = free_slot(BLOCK, BOWL, occupants=())
    occupant = Box(center=(first_x, first_y, 0.75), half_size=BLOCK.half_size)

    _, second_y = free_slot(BLOCK, BOWL, occupants=(occupant,))

    assert abs(second_y - first_y) >= 2 * BLOCK.half_size[1]
    assert abs(second_y) + BLOCK.half_size[1] <= BOWL.half_size[1]


def test_free_slot_reports_a_full_container() -> None:
    first_x, first_y = free_slot(BLOCK, BOWL, occupants=())
    one = Box(center=(first_x, first_y, 0.75), half_size=BLOCK.half_size)
    second_x, second_y = free_slot(BLOCK, BOWL, occupants=(one,))
    two = Box(center=(second_x, second_y, 0.75), half_size=BLOCK.half_size)

    with pytest.raises(ValueError, match="full"):
        free_slot(BLOCK, BOWL, occupants=(one, two))


def test_object_hovering_above_a_container_is_not_in_it() -> None:
    hovering = Box(center=(0.30, 0.00, 0.95), half_size=(0.025, 0.025, 0.03))

    assert classify_relation(hovering, BOWL) is not Relation.IN
