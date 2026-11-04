"""A small learned model that predicts whether a Task A run will succeed.

It's logistic regression, written with NumPy alone: given a layout and a way of doing the
task, it predicts the chance of success, so the robot can pick the way most likely to work
before it moves. It isn't a world model: it predicts only success, not how the scene changes
step by step. It learns from saved simulator runs.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

MODES = ("nearest", "two")


def features(starts: Sequence[Sequence[float]], tower: Sequence[float], mode: str) -> np.ndarray:
    """Numbers describing a layout and a way of doing it, in centimetres on the table.

    Positions are (forward from the person's edge, sideways with plus on their left).
    """
    points = np.array(starts, dtype=float)
    tower_point = np.array(tower, dtype=float)
    two = 1.0 if mode == "two" else 0.0
    gaps = np.linalg.norm(points - tower_point, axis=1)
    layout = np.array(
        [
            tower_point[0],
            abs(tower_point[1]),
            float(np.mean(points[:, 0])),
            float(np.mean(np.abs(points[:, 1]))),
            float(np.max(np.abs(points[:, 1]))),
            float(np.mean(gaps)),
            # How many glasses start on the other side of the robot from the tower.
            float(np.sum(np.sign(points[:, 1]) != np.sign(tower_point[1]))),
        ]
    )
    return np.concatenate(([1.0, two], layout, two * layout)) / SCALES


SCALES = np.array([1.0, 1.0, *([30.0] * 6), 3.0, *([30.0] * 6), 3.0])


@dataclass(frozen=True)
class OutcomeModel:
    weights: np.ndarray

    def chance(self, starts, tower, mode: str) -> float:
        return float(1 / (1 + np.exp(-features(starts, tower, mode) @ self.weights)))

    def best_mode(self, starts, tower) -> str:
        return max(MODES, key=lambda mode: self.chance(starts, tower, mode))


def table_cm(robot_point, near_edge: float) -> tuple[float, float]:
    """A robot-frame position in metres, as (forward from the edge, sideways) in centimetres."""
    return ((float(robot_point[0]) - near_edge) * 100, float(robot_point[1]) * 100)


def train(
    rows: Sequence[tuple[np.ndarray, bool]],
    rounds: int = 4000,
    step: float = 0.5,
    shrink: float = 0.01,
) -> OutcomeModel:
    """Fit by gradient descent, with a little shrinkage so a few runs can't dominate."""
    x = np.array([row[0] for row in rows])
    y = np.array([1.0 if row[1] else 0.0 for row in rows])
    weights = np.zeros(x.shape[1])
    for _ in range(rounds):
        predicted = 1 / (1 + np.exp(-x @ weights))
        gradient = x.T @ (predicted - y) / len(y) + shrink * weights
        weights = weights - step * gradient
    return OutcomeModel(weights)
