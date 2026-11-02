"""Measure how reliably the robot works with glasses and towers at random table spots.

Each trial is a complete task, with a pinned pelvis and exact simulator positions. It tests
what the robot can physically do, not how well a video is read.

Positions are in centimetres: forward from the person's edge of the table, and sideways from
its centre line, where plus is the person's left.

PYTHONPATH=. .venv/bin/python scripts/area_test.py triangle --trials 30 --seed 8123
PYTHONPATH=. .venv/bin/python scripts/area_test.py flip --trials 30 --seed 9001
PYTHONPATH=. .venv/bin/python scripts/area_test.py triangle --spread whole --arm nearest --trials 30 --seed 8500
"""

import argparse
import dataclasses
import json
import logging
from multiprocessing import Pool

import numpy as np

from src.reasoning.fit_layout import nearest_arm
from src.simulation.cup_flip import build_flipped_tower, load_flip_config
from src.simulation.glass_task import build_glass_triangle
from src.simulation.robot_env import RobotEnv, load_motion_config, motion_for_arm
from src.simulation.scene import ROOT, build_model, load_scene_config
from src.verification.triangle import load_triangle_config

Spot = tuple[float, float]
Box = tuple[tuple[float, float], tuple[float, float]]
TRIANGLE_CONFIG = ROOT / "configs/cups_real.yaml"
FLIP_CONFIG = ROOT / "configs/flip_real.yaml"
OUTPUT = ROOT / "data/processed/area_test"
NEAR_EDGE = 0.16
WORKERS = 3
GLASS_GAP_CM = 12.0
TOWER_GAP_CM = 15.0
TRIANGLE_STARTS: Box = ((12, 34), (-30, 0))
TRIANGLE_TOWER: Box = ((24, 36), (-24, 12))
WHOLE_TABLE_STARTS: Box = ((12, 34), (-30, 30))
WHOLE_TABLE_TOWER: Box = ((24, 36), (-24, 24))
LEFT_SIDE_STARTS: Box = ((12, 34), (0, 36))
LEFT_SIDE_TOWER: Box = ((24, 36), (-12, 24))
FLIP_STARTS: Box = ((13, 19), (-10, 14))
FLIP_FOURTH: Box = ((23, 28), (-10, 4))
FLIP_TOWER: Box = ((20, 27), (-31, -26))
TRIANGLE_SPREADS: dict[str, tuple[Box, Box]] = {
    "usual": (TRIANGLE_STARTS, TRIANGLE_TOWER),
    "whole": (WHOLE_TABLE_STARTS, WHOLE_TABLE_TOWER),
    "left": (LEFT_SIDE_STARTS, LEFT_SIDE_TOWER),
}
ORDER = ("cup_a", "cup_b", "cup_c")
FOURTH = "cup_d"


def _in_simulator(spot: Spot) -> Spot:
    return (NEAR_EDGE + spot[0] / 100, spot[1] / 100)


def _far_enough(first: Spot, second: Spot, gap: float) -> bool:
    return bool(np.hypot(first[0] - second[0], first[1] - second[1]) >= gap)


def _random_spot(rng: np.random.Generator, box: Box) -> Spot:
    return (float(rng.uniform(*box[0])), float(rng.uniform(*box[1])))


def random_layout(
    rng: np.random.Generator, flip: bool, spread: str = "usual"
) -> tuple[tuple[Spot, ...], Spot]:
    """Choose glass starts and a tower, with room for the hand between them all."""
    while True:
        if flip:
            tower = _random_spot(rng, FLIP_TOWER)
            starts = (
                *(_random_spot(rng, FLIP_STARTS) for _ in ORDER),
                _random_spot(rng, FLIP_FOURTH),
            )
        else:
            start_box, tower_box = TRIANGLE_SPREADS[spread]
            tower = _random_spot(rng, tower_box)
            starts = tuple(_random_spot(rng, start_box) for _ in ORDER)
        apart = all(
            _far_enough(starts[i], starts[j], GLASS_GAP_CM)
            for i in range(len(starts))
            for j in range(i)
        )
        if apart and all(_far_enough(s, tower, TOWER_GAP_CM) for s in starts):
            return starts, tower


def run_trial(job: tuple[tuple[Spot, ...], Spot, str]) -> dict:
    starts, tower, arm_choice = job
    logging.disable(logging.WARNING)
    flip = len(starts) > len(ORDER)
    path = FLIP_CONFIG if flip else TRIANGLE_CONFIG
    scene = load_scene_config(path)
    names = (*ORDER, FOURTH) if flip else ORDER
    layout = {
        name: _in_simulator(spot) for name, spot in zip(names, starts, strict=True)
    }
    arm = "right"
    if arm_choice == "nearest":
        arm = nearest_arm(
            [np.array(_in_simulator(spot)) for spot in starts],
            np.array(_in_simulator(tower)),
        )
    triangle = dataclasses.replace(
        load_triangle_config(path),
        center=_in_simulator(tower),
        first_on_left=arm == "right",
    )
    motion = motion_for_arm(load_motion_config(path), arm)
    env = RobotEnv(build_model(scene, layout), motion)
    if flip:
        result = build_flipped_tower(
            env,
            scene,
            triangle,
            load_flip_config(path),
            ORDER,
            FOURTH,
            build_glass_triangle,
        )
    else:
        result = build_glass_triangle(env, scene, triangle, ORDER)
    return {
        "starts": starts,
        "tower": tower,
        "arm": arm,
        "success": result.success,
        "failures": list(result.failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("triangle", "flip"))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--spread",
        choices=tuple(TRIANGLE_SPREADS),
        default="usual",
        help="for a triangle, where the glasses and tower can be",
    )
    parser.add_argument("--arm", choices=("right", "nearest"), default="right")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    jobs = [
        (*random_layout(rng, args.task == "flip", args.spread), args.arm)
        for _ in range(args.trials)
    ]
    with Pool(WORKERS) as pool:
        report = pool.map(run_trial, jobs, chunksize=1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    label = f"{args.task}_{args.arm}" if args.arm != "right" else args.task
    label = f"{label}_{args.spread}" if args.spread != "usual" else label
    (OUTPUT / f"{label}_seed_{args.seed}.json").write_text(
        json.dumps(report, indent=2)
    )
    for trial in report:
        if not trial["success"]:
            print("FAIL", trial)
    passed = sum(trial["success"] for trial in report)
    print(f"Result: {passed} of {len(report)} passed")


if __name__ == "__main__":
    main()
