"""Measure how reliably the G1 builds the cup triangle, from random starting positions.

The plan here is written by hand: this tests the physics only, not video understanding.
"""

import argparse
import json
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.simulation.cup_task import build_triangle
from src.simulation.glass_task import build_glass_triangle
from src.simulation.recorder import video_recorder
from src.simulation.robot_env import RobotEnv, load_motion_config
from src.simulation.scene import ROOT, build_model, load_scene_config
from src.verification.triangle import load_triangle_config

DEFAULT_CONFIG = "configs/cups.yaml"
REAL_GLASS_CONFIG = "configs/cups_real.yaml"
START_X = (0.29, 0.38)
START_Y = (-0.34, -0.02)
MIN_SPACING = 0.12
WORKERS = 3
logger = logging.getLogger(__name__)


def random_layout(seed: int, names: list[str]) -> dict[str, tuple[float, float]]:
    """Scatter the cups within the area the arm reaches, leaving room for the hand."""
    rng = np.random.default_rng(seed)
    while True:
        points = np.column_stack(
            (
                rng.uniform(*START_X, len(names)),
                rng.uniform(*START_Y, len(names)),
            )
        )
        gaps = [
            np.linalg.norm(points[i] - points[j])
            for i in range(len(names))
            for j in range(i)
        ]
        if min(gaps) >= MIN_SPACING:
            return {
                name: (float(x), float(y))
                for name, (x, y) in zip(names, points, strict=True)
            }


def _output_folder(config_name: str) -> Path:
    return ROOT / "data/processed" / Path(config_name).stem


def run_trial(job: tuple[int, bool, str]) -> dict:
    seed, record, config_name = job
    config_path = ROOT / config_name
    config = load_scene_config(config_path)
    motion = load_motion_config(config_path)
    triangle = load_triangle_config(config_path)
    names = [item.name for item in config.objects]
    layout = random_layout(seed, names)
    model = build_model(config, layout)
    order = tuple(np.random.default_rng(seed + 1).permutation(names))
    build = build_glass_triangle if config_name == REAL_GLASS_CONFIG else build_triangle
    if record:
        with video_recorder(model, _output_folder(config_name) / f"seed_{seed}.mp4") as on_frame:
            result = build(
                RobotEnv(model, motion, on_frame=on_frame), config, triangle, order
            )
    else:
        result = build(RobotEnv(model, motion), config, triangle, order)
    return {
        "seed": seed,
        "success": result.success,
        "picked": list(result.picked),
        "failures": list(result.failures),
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
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
    path = output / f"report_seeds_{args.first_seed}_x{args.trials}.json"
    path.write_text(json.dumps(report, indent=2))
    for trial in report:
        status = "PASS" if trial["success"] else "FAIL"
        print(trial["seed"], status, trial["picked"], trial["failures"])
    passed = sum(trial["success"] for trial in report)
    print(f"Result: {passed} of {len(report)} passed")


if __name__ == "__main__":
    main()
