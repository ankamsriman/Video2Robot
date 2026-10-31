"""Work out the plan directly from where objects moved in the video, with no AI model.

An object that ends up somewhere new was moved by the demonstrator. When it moved gives
the order of the steps, and where it landed gives the relation.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from src.schemas.plan import Plan, Step
from src.verification.relations import Box, classify_relation

NOMINAL_TABLE_HEIGHT = 0.0
Position = tuple[float, float]


@dataclass(frozen=True)
class ItemInfo:
    name: str
    size: tuple[float, float, float]
    is_container: bool


@dataclass(frozen=True)
class ObserverConfig:
    move_threshold: float
    stable_seconds: float


@dataclass(frozen=True)
class DemoObservation:
    plan: Plan
    start_positions: Mapping[str, Position]
    final_positions: Mapping[str, Position]


def _settled_position(
    track: np.ndarray, frames: int, name: str, label: str
) -> Position:
    """Take the median of the seen frames, which resists brief tracking errors."""
    seen = track[~np.isnan(track[:, 0])]
    if not len(seen):
        raise ValueError(f"'{name}' was not visible at the {label} of the video")
    window = seen[:frames] if label == "start" else seen[-frames:]
    x, y = np.median(window, axis=0)
    return (float(x), float(y))


def _first_departure(track: np.ndarray, start: Position, threshold: float) -> int:
    """Find when the object first left its start, counting being hidden as leaving."""
    distance = np.linalg.norm(track - np.array(start), axis=1)
    away = np.isnan(distance) | (distance > threshold)
    first_seen = int(np.argmax(~np.isnan(distance)))
    settled_away = np.nonzero(distance[first_seen:] > threshold)[0]
    if not len(settled_away):
        return len(track)
    last_away_run = first_seen + int(settled_away[0])
    while last_away_run > first_seen and away[last_away_run - 1]:
        last_away_run -= 1
    return last_away_run


def _box(item: ItemInfo, position: Position) -> Box:
    half = tuple(length / 2 for length in item.size)
    return Box(center=(*position, NOMINAL_TABLE_HEIGHT), half_size=half)


def _reference_for(
    moved: ItemInfo,
    position: Position,
    items: tuple[ItemInfo, ...],
    finals: Mapping[str, Position],
) -> ItemInfo:
    """Prefer a container the object landed in, otherwise use the nearest item."""
    others = [item for item in items if item.name != moved.name]
    for item in others:
        if not item.is_container:
            continue
        offset = np.abs(np.array(position) - np.array(finals[item.name]))
        if (offset <= np.array(item.size[:2]) / 2).all():
            return item
    return min(
        others,
        key=lambda item: float(
            np.linalg.norm(np.array(position) - np.array(finals[item.name]))
        ),
    )


def observe_demo(
    tracks: Mapping[str, np.ndarray],
    items: tuple[ItemInfo, ...],
    fps: float,
    config: ObserverConfig,
) -> DemoObservation:
    """Turn per-frame table positions of every item into a plan and start positions."""
    frames = max(int(config.stable_seconds * fps), 1)
    starts = {
        item.name: _settled_position(tracks[item.name], frames, item.name, "start")
        for item in items
    }
    finals = {
        item.name: _settled_position(tracks[item.name], frames, item.name, "end")
        for item in items
    }
    moves: list[tuple[int, Step]] = []
    for item in items:
        travelled = np.linalg.norm(
            np.array(finals[item.name]) - np.array(starts[item.name])
        )
        if item.is_container or travelled < config.move_threshold:
            continue
        reference = _reference_for(item, finals[item.name], items, finals)
        relation = classify_relation(
            _box(item, finals[item.name]),
            _box(reference, finals[reference.name]),
            container=reference.is_container,
        )
        step = Step(object=item.name, reference=reference.name, relation=relation)
        moves.append(
            (
                _first_departure(
                    tracks[item.name], starts[item.name], config.move_threshold
                ),
                step,
            )
        )
    if not moves:
        raise ValueError("No object moved in this video, so there's no task to copy")
    ordered = tuple(step for _, step in sorted(moves, key=lambda move: move[0]))
    return DemoObservation(Plan(steps=ordered), starts, finals)
