"""Measure where the G1 really works, with a full physics pick and place at each spot.

Tests object starting positions and tray positions separately, and saves both maps.
"""

import json
import logging
from multiprocessing import Pool

import numpy as np

from src.schemas.plan import Plan, Relation, Step
from src.simulation.primitives import execute_plan
from src.simulation.robot_env import RobotEnv, load_motion_config
from src.simulation.scene import ROOT, build_model, load_scene_config

OUTPUT = ROOT / "data/processed/workspace_map.json"
OUT_OF_THE_WAY = {"lemon": (0.65, 0.38), "lime": (0.65, 0.30)}
OBJECT_X = np.arange(0.26, 0.43, 0.04)
OBJECT_Y = np.arange(-0.42, 0.15, 0.07)
TRAY_X = np.arange(0.42, 0.59, 0.04)
TRAY_Y = np.arange(-0.35, 0.16, 0.07)
FIXED_TRAY = {"box": (0.50, -0.21), "bowl": (0.62, 0.20)}
FIXED_OBJECT = (0.33, -0.16)
WORKERS = 3
PLAN = Plan(steps=[Step(object="apple", reference="box", relation=Relation.IN)])
logger = logging.getLogger(__name__)


def run_case(case: tuple[str, float, float]) -> dict:
    kind, x, y = case
    config = load_scene_config()
    positions = {**OUT_OF_THE_WAY, **FIXED_TRAY, "apple": FIXED_OBJECT}
    positions["apple" if kind == "object" else "box"] = (x, y)
    env = RobotEnv(build_model(config, positions), load_motion_config())
    result = execute_plan(env, config, PLAN)
    return {"kind": kind, "x": x, "y": y, "success": result.success}


def _overlaps_tray(x: float, y: float) -> bool:
    tray_x, tray_y = FIXED_TRAY["box"]
    return abs(x - tray_x) < 0.11 and abs(y - tray_y) < 0.125


def _overlaps_object(x: float, y: float) -> bool:
    return abs(x - FIXED_OBJECT[0]) < 0.11 and abs(y - FIXED_OBJECT[1]) < 0.125


def _cases() -> list[tuple[str, float, float]]:
    objects = [
        ("object", float(x), float(y))
        for x in OBJECT_X
        for y in OBJECT_Y
        if not _overlaps_tray(x, y)
    ]
    trays = [
        ("tray", float(x), float(y))
        for x in TRAY_X
        for y in TRAY_Y
        if not _overlaps_object(x, y)
    ]
    return objects + trays


def _print_map(results: list[dict], kind: str) -> None:
    chosen = [item for item in results if item["kind"] == kind]
    ys = sorted({item["y"] for item in chosen})
    logger.info("\n%s position map (# passed, . failed), rows x, columns y", kind)
    logger.info("x\\y    %s", " ".join(f"{y:+.2f}" for y in ys))
    for x in sorted({item["x"] for item in chosen}, reverse=True):
        row = {item["y"]: item["success"] for item in chosen if item["x"] == x}
        cells = [
            "  #  " if row.get(y) else "  .  " if y in row else "     " for y in ys
        ]
        logger.info("%.2f   %s", x, " ".join(cells))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with Pool(WORKERS) as pool:
        results = pool.map(run_case, _cases(), chunksize=1)
    OUTPUT.write_text(json.dumps(results, indent=2))
    _print_map(results, "object")
    _print_map(results, "tray")
    passed = sum(item["success"] for item in results)
    logger.info("\nPassed %d of %d", passed, len(results))


if __name__ == "__main__":
    logging.disable(logging.NOTSET)
    main()
