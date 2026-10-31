"""Score a predicted plan against the correct one, using SeeDo's three metrics.

SSR uses the longest common subsequence, as SeeDo's released evaluation code does. The
pointer method printed in the paper can wrongly match steps out of order.
"""

from collections import Counter
from enum import Enum

from src.schemas.plan import Plan, Step


class ErrorType(str, Enum):
    VISION = "vision"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"


def tsr(predicted: Plan, truth: Plan) -> int:
    """Task success: 1 only if every step matches, in order."""
    return int(predicted.steps == truth.steps)


def _final_state(plan: Plan) -> dict[str, Step]:
    return {step.object: step for step in plan.steps}


def fsr(predicted: Plan, truth: Plan) -> int:
    """Final-state success: 1 if every object ends in the right place, in any order."""
    return int(_final_state(predicted) == _final_state(truth))


def ssr(predicted: Plan, truth: Plan) -> float:
    """Step success: the longest run of correct steps kept in order, as a fraction."""
    best = [0] * (len(truth.steps) + 1)
    for step in predicted.steps:
        previous_diagonal = 0
        for index, expected in enumerate(truth.steps, start=1):
            saved = best[index]
            if step == expected:
                best[index] = previous_diagonal + 1
            else:
                best[index] = max(best[index], best[index - 1])
            previous_diagonal = saved
    return best[-1] / len(truth.steps)


def classify_error(predicted: Plan, truth: Plan) -> ErrorType | None:
    """Name the main error type, based on the first step that differs.

    Unlike SeeDo's script, a known object appearing at the wrong time counts as a
    temporal error rather than a vision error, because it was recognised correctly.
    """
    if predicted.steps == truth.steps:
        return None
    if Counter(predicted.steps) == Counter(truth.steps):
        return ErrorType.TEMPORAL
    known_objects = {step.object for step in truth.steps}
    for ours, expected in zip(predicted.steps, truth.steps, strict=False):
        if ours == expected:
            continue
        if ours.object == expected.object:
            return ErrorType.SPATIAL
        return ErrorType.TEMPORAL if ours.object in known_objects else ErrorType.VISION
    return ErrorType.TEMPORAL
