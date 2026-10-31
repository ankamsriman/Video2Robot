"""The retry loop: try a step, measure the result, adjust, and try again."""

from dataclasses import replace

from src.schemas.plan import Plan, Relation, Step
from src.simulation.primitives import StepResult
from src.simulation.robot_env import load_motion_config
from src.verification.closed_loop import failure_reason, next_motion, run_closed_loop

APPLE = Step(object="apple", reference="box", relation=Relation.IN)
LEMON = Step(object="lemon", reference="box", relation=Relation.IN)
PLAN = Plan(steps=(APPLE, LEMON))
MOTION = load_motion_config()


def _ok(step: Step) -> StepResult:
    return StepResult(step, picked=True, reached=True, measured_relation=step.relation)


def _not_picked(step: Step) -> StepResult:
    return StepResult(step, picked=False, reached=False, measured_relation=None)


def _wrong_place(step: Step) -> StepResult:
    return StepResult(
        step, picked=True, reached=True, measured_relation=Relation.TO_THE_LEFT
    )


class ScriptedRobot:
    """Plays back chosen outcomes, in order, for each step."""

    def __init__(self, outcomes: dict[str, list]) -> None:
        self.outcomes = {name: list(items) for name, items in outcomes.items()}
        self.motions = []
        self.final = {APPLE: APPLE.relation, LEMON: LEMON.relation}

    def run_step(self, step: Step, motion: object) -> StepResult:
        self.motions.append(motion)
        queue = self.outcomes.get(step.object, [])
        return (queue.pop(0) if queue else _ok)(step)

    def measure(self, step: Step) -> Relation:
        return self.final[step]


def test_a_clean_run_takes_one_attempt_per_step() -> None:
    robot = ScriptedRobot({})

    result = run_closed_loop(robot, PLAN, MOTION, max_attempts=3)

    assert result.success
    assert result.attempts == {"apple": 1, "lemon": 1}
    assert result.total_attempts == 2


def test_a_failed_step_is_retried_and_counted() -> None:
    robot = ScriptedRobot({"apple": [_not_picked, _wrong_place]})

    result = run_closed_loop(robot, PLAN, MOTION, max_attempts=3)

    assert result.success
    assert result.attempts == {"apple": 3, "lemon": 1}
    assert result.failures == ("pick", "placement")


def test_it_gives_up_after_the_attempt_limit() -> None:
    robot = ScriptedRobot({"apple": [_not_picked] * 5})

    result = run_closed_loop(robot, PLAN, MOTION, max_attempts=2)

    assert not result.success
    assert result.attempts["apple"] == 2
    assert "lemon" not in result.attempts


def test_a_retry_uses_adjusted_motion() -> None:
    robot = ScriptedRobot({"apple": [_not_picked]})

    run_closed_loop(robot, PLAN, MOTION, max_attempts=2)

    assert robot.motions[0] == MOTION
    assert robot.motions[1] != MOTION


def test_motion_is_reset_for_the_next_step() -> None:
    robot = ScriptedRobot({"apple": [_not_picked]})

    run_closed_loop(robot, PLAN, MOTION, max_attempts=2)

    assert robot.motions[2] == MOTION


def test_an_earlier_object_knocked_out_of_place_is_redone() -> None:
    robot = ScriptedRobot({})
    robot.final = {APPLE: Relation.TO_THE_LEFT, LEMON: LEMON.relation}

    result = run_closed_loop(robot, PLAN, MOTION, max_attempts=3)

    assert result.attempts["apple"] == 2


def test_failure_reasons_describe_what_went_wrong() -> None:
    assert failure_reason(_not_picked(APPLE)) == "pick"
    assert failure_reason(_wrong_place(APPLE)) == "placement"


def test_pick_failures_grip_harder_and_placement_failures_slow_down() -> None:
    grip = next_motion(MOTION, "pick", 1)
    slow = next_motion(MOTION, "placement", 1)

    assert grip.grip_squeeze > MOTION.grip_squeeze
    assert slow.hand_speed < MOTION.hand_speed


def test_step_succeeds_when_the_object_landed_correctly_even_if_the_hand_fell_short() -> (
    None
):
    short = StepResult(APPLE, picked=True, reached=False, measured_relation=Relation.IN)

    assert short.success


def test_adjustments_grow_with_each_attempt() -> None:
    first = next_motion(MOTION, "pick", 1)
    second = next_motion(MOTION, "pick", 2)

    assert second.grip_squeeze > first.grip_squeeze
    assert replace(MOTION) == MOTION
