"""Measure what the retry loop adds, by making the task harder on purpose.

Each condition runs the same setups twice: once with one attempt per step, and once with
retries. Saves a Markdown table to data/processed/stress_retry.md.
"""

import json
import logging
from dataclasses import dataclass
from multiprocessing import Pool

import mujoco
import numpy as np

from scripts.phase1_gate import randomised_positions
from src.schemas.plan import load_plan
from src.simulation.robot_env import RobotEnv, load_motion_config
from src.simulation.scene import ROOT, build_model, load_scene_config
from src.verification.closed_loop import SimulatedRobot, run_closed_loop

OUTPUT = ROOT / "data/processed"
PLAN_PATH = ROOT / "data/labels/sample_plan.json"
TRIALS = 12
FIRST_SEED = 700
MAX_ATTEMPTS = 4
WORKERS = 3


@dataclass(frozen=True)
class Condition:
    name: str
    mass: float = 0.08
    friction: float = 1.5
    finger_force: float = 1.0
    yaw: float = 0.0


CONDITIONS = (
    Condition("Normal"),
    Condition("Heavy, 300 g", mass=0.30),
    Condition("Slippery", friction=0.25),
    Condition("Weak grip", finger_force=0.3),
    Condition("Rotated 30 degrees", yaw=np.radians(30)),
    Condition("Heavy and slippery", mass=0.30, friction=0.25),
)


def _apply(env: RobotEnv, config, condition: Condition) -> None:
    model, data = env.model, env.data
    for item in config.objects:
        body = model.body(item.name).id
        ratio = condition.mass / model.body_mass[body]
        model.body_mass[body] *= ratio
        model.body_inertia[body] *= ratio
        model.geom(f"{item.name}_geom").friction[0] = condition.friction
        if condition.yaw:
            address = model.joint(f"{item.name}_free").qposadr[0]
            half = condition.yaw / 2
            data.qpos[address + 3 : address + 7] = (
                np.cos(half),
                0.0,
                0.0,
                np.sin(half),
            )
    for index in range(model.nu):
        if "_hand_" in model.actuator(index).name:
            limit = condition.finger_force
            model.actuator_forcerange[index] = (-limit, limit)
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)


def run_case(job: tuple[Condition, int, int]) -> dict:
    condition, seed, attempts = job
    config = load_scene_config()
    motion = load_motion_config()
    model = build_model(config, randomised_positions(config, seed))
    env = RobotEnv(model, motion)
    _apply(env, config, condition)
    outcome = run_closed_loop(
        SimulatedRobot(env, config), load_plan(PLAN_PATH), motion, attempts
    )
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
