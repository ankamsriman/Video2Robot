"""Check each step of a pick and place, so a failure says where it happened.

A task is many small steps: reach the glass, grip it, lift it, carry it, set it down, let
go. The final check only says whether the tower is right. These checks run after each step,
from exact simulator positions, so a failed run also says which step went wrong. They only
measure: nothing here changes what the robot does.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

# Generous enough that a step that went fine always passes, tight enough to catch real slips.
REACH_TOLERANCE = 0.02
MAX_HOLD_DISTANCE = 0.06
MIN_LIFT = 0.05
PLACE_TOLERANCE = 0.01
MAX_TILT_DEGREES = 10.0
STILL_TOLERANCE = 0.01


@dataclass(frozen=True)
class StepCheck:
    """One step's result for one glass, with the measurement behind it."""

    glass: str
    step: str
    passed: bool
    detail: str

    def __post_init__(self) -> None:
        # Comparisons of NumPy numbers give NumPy booleans, which report files can't hold.
        object.__setattr__(self, "passed", bool(self.passed))


def _cm(metres: float) -> str:
    return f"{metres * 100:.1f} cm"


def hand_reached(glass: str, hand: np.ndarray, target: np.ndarray) -> StepCheck:
    """The hand got to where it grips the glass."""
    miss = float(np.linalg.norm(hand - target))
    return StepCheck(glass, "reach", miss <= REACH_TOLERANCE, f"hand {_cm(miss)} off")


def glass_held(
    glass: str, step: str, position: np.ndarray, hand: np.ndarray
) -> StepCheck:
    """The glass is still in the hand after `step`."""
    gap = float(np.linalg.norm(position - hand))
    return StepCheck(
        glass, step, gap <= MAX_HOLD_DISTANCE, f"glass {_cm(gap)} from the hand"
    )


def glass_lifted(glass: str, start_height: float, height: float) -> StepCheck:
    """The glass rose off the table with the hand."""
    rise = height - start_height
    return StepCheck(glass, "lift", rise >= MIN_LIFT, f"raised {_cm(rise)}")


def glass_at(
    glass: str, step: str, position: np.ndarray, target: np.ndarray
) -> StepCheck:
    """The glass is over its target, measured across the table only."""
    miss = float(np.linalg.norm(np.asarray(position)[:2] - np.asarray(target)[:2]))
    return StepCheck(
        glass, step, miss <= PLACE_TOLERANCE, f"{_cm(miss)} from the target"
    )


def glass_upright(
    glass: str, step: str, up: np.ndarray, flipped: bool = False
) -> StepCheck:
    """The glass stands straight, or straight upside down if it was meant to be flipped."""
    vertical = -up[2] if flipped else up[2]
    tilt = float(np.degrees(np.arccos(np.clip(vertical, -1.0, 1.0))))
    return StepCheck(
        glass, step, tilt <= MAX_TILT_DEGREES, f"leaning {tilt:.0f} degrees"
    )


def others_still(
    glass: str,
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
) -> StepCheck:
    """Moving one glass left the others where they were."""
    moved = {
        name: float(np.linalg.norm(after[name] - position))
        for name, position in before.items()
    }
    knocked = [name for name, distance in moved.items() if distance > STILL_TOLERANCE]
    detail = (
        "others untouched"
        if not knocked
        else ", ".join(f"{name} moved {_cm(moved[name])}" for name in knocked)
    )
    return StepCheck(glass, "others still", not knocked, detail)


def first_failure(checks: Iterable[StepCheck]) -> str | None:
    """Where the run first went wrong, like 'cup_b: carry', or None if every step passed."""
    return next((f"{c.glass}: {c.step}" for c in checks if not c.passed), None)


def step_summary(checks: Iterable[StepCheck]) -> list[dict]:
    """The checks as plain data, for a report file."""
    return [
        {"glass": c.glass, "step": c.step, "passed": c.passed, "detail": c.detail}
        for c in checks
    ]


def step_tally(runs: Iterable[Mapping]) -> dict[str, int]:
    """How many failed runs first went wrong at each step, most common first.

    The glass name is dropped, so 'cup_a: carry' and 'cup_b: carry' both count as 'carry'.
    A failed run with no failed step (only the final check failed) counts as 'final check'.
    """
    steps = Counter(
        run["failed_at"].split(": ")[-1] if run.get("failed_at") else "final check"
        for run in runs
        if not run["success"]
    )
    return dict(steps.most_common())


def tally_lines(runs: Iterable[Mapping]) -> list[str]:
    """Where failed runs went wrong, as lines for a test script's summary."""
    return [
        f"  first failed step: {step:<14} {count}"
        for step, count in step_tally(runs).items()
    ]
