"""Measure what the retry loop adds, by making the task harder on purpose.

Each condition runs the same setups twice: once with one attempt per step, and once with
retries. Saves a Markdown table to data/processed/stress_retry.md.
"""

import json
import logging
from dataclasses import dataclass, replace
from multiprocessing import Pool

import mujoco
import numpy as np

from scripts.phase1_gate import randomised_positions
from src.schemas.plan import Plan, Step, load_plan
from src.simulation.robot_env import RobotEnv, load_motion_config
from src.simulation.scene import ROOT, build_model, load_scene_config
from src.verification.closed_loop import SimulatedRobot, run_closed_loop

OUTPUT = ROOT / "data/processed"
PLAN_PATH = ROOT / "data/labels/sample_plan.json"
TRIALS = 8
FIRST_SEED = 800
KNOCKED_TO = (0.36, 0.10)
MAX_ATTEMPTS = 4
WORKERS = 3
NOTE = (
    "Even with one attempt per step, the final recheck still runs. So the knocked-object "
    "row isn't a true no-loop comparison: with no recheck, that object stays out of place."
)


@dataclass(frozen=True)
class Condition:
    name: str
    mass: float = 0.08
    friction: float | None = None
    finger_force: float = 1.0
    yaw: float = 0.0
    knock_after_first_step: bool = False


CONDITIONS = (
    Condition("Normal"),
    Condition("Heavy, 300 g", mass=0.30),
    Condition("Rotated 30 degrees", yaw=np.radians(30)),
    Condition("Slippery, friction 0.3", friction=0.3),
    Condition("Very slippery, friction 0.15", friction=0.15),
    Condition("Weak grip, 0.5 N m", finger_force=0.5),
    Condition("Weak grip, 0.3 N m", finger_force=0.3),
    Condition("Very weak grip, 0.15 N m", finger_force=0.15),
    Condition("Object knocked out after placing", knock_after_first_step=True),
)


class KnockingRobot(SimulatedRobot):
    """Moves the first placed object off target, as a bump from a person might."""

    def __init__(self, env: RobotEnv, config, plan: Plan) -> None:
        super().__init__(env, config)
        self._env_ref = env
        self._name = plan.steps[0].object
        self._knocked = False

    def run_step(self, step: Step, motion):
        result = super().run_step(step, motion)
        if step.object == self._name and not self._knocked and result.success:
            self._knocked = True
            address = self._env_ref.model.joint(f"{self._name}_free").qposadr[0]
            self._env_ref.data.qpos[address : address + 2] = KNOCKED_TO
            self._env_ref.advance(0.5)
        return result


def _apply(env: RobotEnv, config, condition: Condition) -> None:
    model, data = env.model, env.data
    for item in config.objects:
        body = model.body(item.name).id
        ratio = condition.mass / model.body_mass[body]
        model.body_mass[body] *= ratio
        model.body_inertia[body] *= ratio
        if condition.friction is not None:
            # MuJoCo uses the higher friction of two touching surfaces, unless one has
            # a higher priority, so priority makes this object's value count.
            geom = model.geom(f"{item.name}_geom")
            geom.friction[0] = condition.friction
            geom.priority = 1
        if condition.yaw:
            address = model.joint(f"{item.name}_free").qposadr[0]
            half = condition.yaw / 2
            data.qpos[address + 3 : address + 7] = (
                np.cos(half),
                0.0,
                0.0,
                np.sin(half),
            )
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)


def run_case(job: tuple[Condition, int, int]) -> dict:
    condition, seed, attempts = job
    config = load_scene_config()
    motion = replace(load_motion_config(), finger_force_limit=condition.finger_force)
    model = build_model(config, randomised_positions(config, seed))
    env = RobotEnv(model, motion)
    _apply(env, config, condition)
    plan = load_plan(PLAN_PATH)
    robot = (
        KnockingRobot(env, config, plan)
        if condition.knock_after_first_step
        else SimulatedRobot(env, config)
    )
    outcome = run_closed_loop(robot, plan, motion, attempts)
    return {
        "condition": condition.name,
        "seed": seed,
        "limit": attempts,
        "success": outcome.success,
        "attempts": outcome.total_attempts,
        "failures": list(outcome.failures),
    }


def _table(results: list[dict]) -> str:
    lines = [
        f"Each condition ran {TRIALS} setups, with a 3-step plan.",
        "",
        "| Condition | One attempt | With retries | Attempts when it succeeded |",
        "|---|---|---|---|",
    ]
    for condition in CONDITIONS:
        rows = [item for item in results if item["condition"] == condition.name]
        single = [item for item in rows if item["limit"] == 1]
        retry = [item for item in rows if item["limit"] == MAX_ATTEMPTS]
        wins = [item["attempts"] for item in retry if item["success"]]
        mean = f"{np.mean(wins):.1f}" if wins else "none"
        lines.append(
            f"| {condition.name} | {sum(i['success'] for i in single)} of {len(single)} "
            f"| {sum(i['success'] for i in retry)} of {len(retry)} | {mean} |"
        )
    lines += ["", NOTE]
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.disable(logging.WARNING)
    jobs = [
        (condition, seed, limit)
        for condition in CONDITIONS
        for seed in range(FIRST_SEED, FIRST_SEED + TRIALS)
        for limit in (1, MAX_ATTEMPTS)
    ]
    with Pool(WORKERS) as pool:
        results = pool.map(run_case, jobs, chunksize=1)
    (OUTPUT / "stress_retry.json").write_text(json.dumps(results, indent=2))
    table = _table(results)
    (OUTPUT / "stress_retry.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
