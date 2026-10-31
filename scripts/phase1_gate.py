"""Phase 1 gate: run a hand-written 3-step plan with randomised positions, in physics."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.schemas.plan import load_plan
from src.simulation.primitives import ExecutionResult, execute_plan
from src.simulation.recorder import video_recorder
from src.simulation.robot_env import RobotEnv, load_motion_config
from src.simulation.scene import ROOT, SceneConfig, build_model, load_scene_config

JITTER = 0.03
OUTPUT = ROOT / "data/processed/phase1"
logger = logging.getLogger(__name__)


def randomised_positions(
    config: SceneConfig, seed: int
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    return {
        item.name: (
            item.position[0] + float(rng.uniform(-JITTER, JITTER)),
            item.position[1] + float(rng.uniform(-JITTER, JITTER)),
        )
        for item in (*config.objects, *config.containers)
    }


def run_trial(
    config: SceneConfig, plan_path: Path, seed: int, pinned: bool, video: bool
) -> ExecutionResult:
    plan = load_plan(plan_path)
    model = build_model(config, randomised_positions(config, seed), pinned=pinned)
    if not video:
        return execute_plan(RobotEnv(model, load_motion_config()), config, plan)
    base = "pinned" if pinned else "free_base"
    with video_recorder(model, OUTPUT / f"{base}_trial_{seed}.mp4") as on_frame:
        env = RobotEnv(model, load_motion_config(), on_frame=on_frame)
        return execute_plan(env, config, plan)


def summarise(seed: int, result: ExecutionResult) -> dict:
    return {
        "seed": seed,
        "success": result.success,
        "steps": [
            {
                "object": item.step.object,
                "picked": item.picked,
                "reached": item.reached,
                "success": item.success,
            }
            for item in result.steps
        ],
        "final": [relation.value for relation in result.final],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--free-base", action="store_true")
    parser.add_argument("--plan", default=str(ROOT / "data/labels/sample_plan.json"))
    args = parser.parse_args()

    config = load_scene_config()
    report = []
    seeds = range(args.first_seed, args.first_seed + args.trials)
    for seed in seeds:
        result = run_trial(
            config, Path(args.plan), seed, not args.free_base, video=seed == seeds[0]
        )
        report.append(summarise(seed, result))
        logger.info("Trial %d: %s", seed, "PASS" if result.success else "FAIL")

    passes = sum(trial["success"] for trial in report)
    logger.info("Result: %d of %d passed", passes, args.trials)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base = "free_base" if args.free_base else "pinned"
    name = f"report_{base}_seeds_{args.first_seed}.json"
    (OUTPUT / name).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
