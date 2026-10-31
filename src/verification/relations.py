"""Convert between spatial relations and table geometry.

World axes follow the robot's view: +x is away from the robot ("back"), +y is the
robot's left, and +z is up.
"""

from dataclasses import dataclass

from src.schemas.plan import Relation

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Box:
    """An axis-aligned box describing an object's position and size."""

    center: Vec3
    half_size: Vec3


def placement_target(obj: Box, reference: Box, relation: Relation, gap: float) -> Vec3:
    """Return where the centre of `obj` should end up for the given relation."""
    rx, ry, rz = reference.center
    if relation is Relation.IN:
        floor = rz - reference.half_size[2]
        return (rx, ry, floor + obj.half_size[2])
    if relation is Relation.ON_TOP_OF:
        return (rx, ry, rz + reference.half_size[2] + obj.half_size[2])

    x_offset = reference.half_size[0] + obj.half_size[0] + gap
    y_offset = reference.half_size[1] + obj.half_size[1] + gap
    offsets = {
        Relation.AT_THE_BACK_OF: (x_offset, 0.0),
        Relation.IN_FRONT_OF: (-x_offset, 0.0),
        Relation.TO_THE_LEFT: (0.0, y_offset),
        Relation.TO_THE_RIGHT: (0.0, -y_offset),
    }
    dx, dy = offsets[relation]
    return (rx + dx, ry + dy, obj.center[2])


def classify_relation(obj: Box, reference: Box, container: bool = True) -> Relation:
    """Return the dominant relation of `obj` to `reference`, as measured from positions."""
    dx = obj.center[0] - reference.center[0]
    dy = obj.center[1] - reference.center[1]
    inside = abs(dx) <= reference.half_size[0] and abs(dy) <= reference.half_size[1]
    reference_top = reference.center[2] + reference.half_size[2]
    below_rim = obj.center[2] - obj.half_size[2] < reference_top
    if inside and container and below_rim:
        return Relation.IN
    if inside and not container:
        return Relation.ON_TOP_OF
    if abs(dx) >= abs(dy):
        return Relation.AT_THE_BACK_OF if dx > 0 else Relation.IN_FRONT_OF
    return Relation.TO_THE_LEFT if dy > 0 else Relation.TO_THE_RIGHT


def free_slot(
    obj: Box, container: Box, occupants: tuple[Box, ...], wall_margin: float = 0.008
) -> tuple[float, float]:
    """Pick a spot inside `container` that doesn't overlap objects already there.

    A tray has two slots, side by side. Objects never go in the centre, because that
    would leave no room for a second one.
    """
    cx, cy, _ = container.center
    offset = container.half_size[1] - obj.half_size[1] - wall_margin
    for y in (cy - offset, cy + offset):
        overlaps = any(
            abs(y - other.center[1]) < obj.half_size[1] + other.half_size[1]
            and abs(cx - other.center[0]) < obj.half_size[0] + other.half_size[0]
            for other in occupants
        )
        if not overlaps:
            return (cx, y)
    raise ValueError("container is full")
