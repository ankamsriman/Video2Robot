"""Test choosing the arm and standing spot: how reliably does the robot repeat a layout?

Each run keeps a layout's glasses where they are and lets the robot choose where to stand.

  video: wobble the layout read from Task 3's video, by about the reader's own uncertainty.
  wide: random layouts anywhere on the table, with the tower on either side of the glasses.

Setups are only ever checked once, so results use new random seeds.

PYTHONPATH=. .venv/bin/python scripts/stand_test.py video --trials 20 --seed 7100
PYTHONPATH=. .venv/bin/python scripts/stand_test.py wide --trials 30 --seed 7200
"""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from src.reasoning.fit_layout import plan_stand
from src.simulation.cup_flip import build_flipped_tower, flip_for_arm, load_flip_config
from src.simulation.glass_task import build_glass_triangle
from src.simulation.robot_env import RobotEnv, load_motion_config, motion_for_arm
from src.simulation.scene import ROOT, build_model, load_scene_config
from src.verification.triangle import load_triangle_config

CONFIG = ROOT / "configs/flip_real.yaml"
NAMES = ("cup_a", "cup_b", "cup_c", "cup_d")
# Simulator positions read from task3_flip_right_take1, robot at the middle of the table.
VIDEO_STARTS = ((0.32, -0.185), (0.389, -0.251), (0.459, -0.164), (0.486, -0.26))
VIDEO_TOWER = (0.406, 0.051)
WOBBLE = 0.015
# Wide layouts: x runs away from the robot, y to its left, with the table's middle at zero.
WIDE_X = (0.29, 0.49)
# The fourth glass only grips reliably from far enough back: 23 cm or more from the near edge.
# With it anywhere in WIDE_X, 12 of 30 passed (docs/results/stand_wide_first.json).
WIDE_FLIPPED_X = (0.39, 0.49)
WIDE_TOWER_X = (0.36, 0.43)
WIDE_TOWER_SIDEWAYS = (0.18, 0.40)
WIDE_SPREAD = 0.20
TABLE_HALF_WIDTH = 0.40
EDGE_MARGIN = 0.05
GLASS_GAP = 0.09
TOWER_GAP = 0.15
MAX_TRIES = 1000
logger = logging.getLogger("stand_test")


def video_layout(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    starts = np.array(VIDEO_STARTS) + rng.uniform(-WOBBLE, WOBBLE, (4, 2))
    tower = np.array(VIDEO_TOWER) + rng.uniform(-WOBBLE, WOBBLE, 2)
    return starts, tower


def _is_clear(starts: np.ndarray, tower: np.ndarray) -> bool:
    gaps = np.linalg.norm(starts[:, None] - starts[None], axis=-1)
    closest = gaps[np.triu_indices(len(starts), 1)].min()
    return bool(
        closest >= GLASS_GAP
        and np.linalg.norm(starts - tower, axis=1).min() >= TOWER_GAP
    )


def wide_layout(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Glasses in a group anywhere along the table, and the tower off to one side of them."""
    limit = TABLE_HALF_WIDTH - EDGE_MARGIN
    for _ in range(MAX_TRIES):
        middle = rng.uniform(-limit, limit)
        ys = rng.uniform(middle - WIDE_SPREAD / 2, middle + WIDE_SPREAD / 2, 4)
        xs = (*rng.uniform(*WIDE_X, 3), rng.uniform(*WIDE_FLIPPED_X))
        starts = np.column_stack((xs, ys))
        side = rng.choice((-1.0, 1.0))
        tower = np.array(
            [
                rng.uniform(*WIDE_TOWER_X),
                middle + side * rng.uniform(*WIDE_TOWER_SIDEWAYS),
            ]
        )
        on_table = np.abs(ys).max() <= limit and abs(tower[1]) <= limit
        if on_table and _is_clear(starts, tower):
            return starts, tower
    raise RuntimeError("Couldn't find a clear layout")


def run_trial(starts: np.ndarray, tower: np.ndarray, tower_reach: float) -> dict:
    stand = plan_stand(list(starts), tower, tower_reach, TABLE_HALF_WIDTH)
    shift = np.array([0.0, stand.robot_y])
    scene = load_scene_config(CONFIG)
    scene = dataclasses.replace(
        scene, table=dataclasses.replace(scene.table, center_y=-stand.robot_y)
    )
    layout = {
        name: tuple(start - shift) for name, start in zip(NAMES, starts, strict=True)
    }
    triangle = dataclasses.replace(
        load_triangle_config(CONFIG),
        center=tuple(tower - shift),
        first_on_left=stand.arm == "right",
    )
    motion = motion_for_arm(load_motion_config(CONFIG), stand.arm)
    env = RobotEnv(build_model(scene, layout), motion)
    result = build_flipped_tower(
        env,
        scene,
        triangle,
        flip_for_arm(load_flip_config(CONFIG), stand.arm),
        NAMES[:3],
        NAMES[3],
        build_glass_triangle,
    )
    return {
        "arm": stand.arm,
        "robot_y": round(stand.robot_y, 3),
        "starts": [[round(float(v), 3) for v in s] for s in starts],
        "tower": [round(float(v), 3) for v in tower],
        "success": result.success,
        "failures": list(result.failures),
    }


def main() -> None:
    logging.basicConfig(level=logging.ERROR, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("video", "wide"))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = yaml.safe_load(CONFIG.read_text())["flip_areas"]
    tower_reach = abs(sum(raw["tower"]["y"]) / 2)
    make_layout = video_layout if args.kind == "video" else wide_layout
    results = []
    for trial in range(args.trials):
        rng = np.random.default_rng(args.seed + trial)
        starts, tower = make_layout(rng)
        outcome = {"seed": args.seed + trial, **run_trial(starts, tower, tower_reach)}
        results.append(outcome)
        print(json.dumps(outcome), flush=True)
        if args.output:
            args.output.write_text(json.dumps(results, indent=1))
    passed = sum(item["success"] for item in results)
    print(f"Result: {passed} of {len(results)} passed")


if __name__ == "__main__":
    main()
