"""Measure how reliably the G1 builds the triangle and flips a fourth cup onto it.

The motion is programmed and the plan is hand-written. It tests the physics, not video
understanding. The pelvis is pinned.
"""

import argparse
import json
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.simulation.cup_flip import build_flipped_tower, load_flip_config
from src.simulation.cup_task import build_triangle
from src.simulation.glass_task import build_glass_triangle
from src.simulation.recorder import video_recorder
from src.simulation.robot_env import RobotEnv, load_motion_config
from src.simulation.scene import ROOT, build_model, load_scene_config
from src.verification.triangle import load_triangle_config

DEFAULT_CONFIG = "configs/cups_flip.yaml"
REAL_GLASS_CONFIG = "configs/flip_real.yaml"
FLIPPED = "cup_d"
ORDER_CUPS = ("cup_a", "cup_b", "cup_c")
START_X = (0.28, 0.35)
START_Y = (-0.10, 0.14)
FLIPPED_X = (0.40, 0.44)
FLIPPED_Y = (-0.12, 0.06)
MIN_SPACING = 0.11
WORKERS = 3


def random_layout(seed: int) -> dict[str, tuple[float, float]]:
    """Scatter the cups, keeping the fourth where the sideways grip works."""
    rng = np.random.default_rng(seed)
    while True:
        points = np.column_stack(
            (rng.uniform(*START_X, 3), rng.uniform(*START_Y, 3))
        ).tolist() + [[rng.uniform(*FLIPPED_X), rng.uniform(*FLIPPED_Y)]]
        gaps = [
            np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            for i in range(4)
            for j in range(i)
        ]
        if min(gaps) >= MIN_SPACING:
            names = (*ORDER_CUPS, FLIPPED)
            return {
                name: (float(x), float(y))
                for name, (x, y) in zip(names, points, strict=True)
            }


def _output_folder(config_name: str) -> Path:
    if config_name == DEFAULT_CONFIG:
        return ROOT / "data/processed/flip"
    return ROOT / "data/processed" / Path(config_name).stem


def run_trial(job: tuple[int, bool] | tuple[int, bool, str]) -> dict:
    seed, record, *rest = job
    config_name = rest[0] if rest else DEFAULT_CONFIG
    CONFIG = ROOT / config_name
    config = load_scene_config(CONFIG)
    motion = load_motion_config(CONFIG)
    triangle = load_triangle_config(CONFIG)
    flip = load_flip_config(CONFIG)
    model = build_model(config, random_layout(seed))
    order = tuple(np.random.default_rng(seed + 1).permutation(ORDER_CUPS))
    base = build_glass_triangle if config_name == REAL_GLASS_CONFIG else build_triangle
    folder = _output_folder(config_name)
    if record:
        with video_recorder(model, folder / f"flip_seed_{seed}.mp4") as on_frame:
            env = RobotEnv(model, motion, on_frame=on_frame)
            result = build_flipped_tower(
                env, config, triangle, flip, order, FLIPPED, base
            )
    else:
        env = RobotEnv(model, motion)
        result = build_flipped_tower(env, config, triangle, flip, order, FLIPPED, base)
    return {"seed": seed, "success": result.success, "failures": list(result.failures)}


def main() -> None:
    logging.disable(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    seeds = range(args.first_seed, args.first_seed + args.trials)
    jobs = [
        (seed, seed == seeds[0] and not args.no_video, args.config) for seed in seeds
    ]
    with Pool(WORKERS) as pool:
        report = pool.map(run_trial, jobs, chunksize=1)
    output = _output_folder(args.config)
    output.mkdir(parents=True, exist_ok=True)
    name = f"report_seeds_{args.first_seed}_x{args.trials}.json"
    (output / name).write_text(json.dumps(report, indent=2))
    for trial in report:
        status = "PASS" if trial["success"] else "FAIL"
        print(trial["seed"], status, trial["failures"])
    passed = sum(trial["success"] for trial in report)
    print(f"Result: {passed} of {len(report)} passed")


if __name__ == "__main__":
    main()
