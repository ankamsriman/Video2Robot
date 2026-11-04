"""Compare two ways of doing Task 2 on every real Task 2 video.

  A: Claude finds the glasses, and the logistic regression model picks the arms
     (each glass with the arm on its side, or one arm for all).
  C: no Claude; each glass always uses the arm on its own side.

Claude is asked once per video, and its points are saved, so rerunning makes no API calls.
A third setup, Claude with each glass always on its own side's arm, was dropped: on every
real video the model already chose exactly that, so it would repeat A.

PYTHONPATH=. .venv/bin/python scripts/compare_task2_setups.py
"""

import json
import logging
from pathlib import Path

import yaml

from scripts.glass_from_video import (
    CONFIG_PATH,
    VideoReadError,
    glasses_from_model,
    read_video,
    run_robot,
)
from src.perception.glass_tracker import GlassVideoConfig
from src.simulation.scene import ROOT

VIDEOS = (
    "task2_triangle_right_take1",
    *(f"task2_triangle_take{n}" for n in range(2, 9)),
    "task2_triangle_check",
)
POINTS = ROOT / "data/processed/task2_setups/claude_points.json"
OUTPUT = ROOT / "docs/results/task2_setups.json"
SETUPS = {
    "A": {"claude": True, "two_arms": False, "choose_arms": True},
    "C": {"claude": False, "two_arms": True, "choose_arms": False},
}


def _claude_points(video: Path, saved: dict) -> list:
    if video.stem not in saved:
        saved[video.stem] = glasses_from_model(video)
        POINTS.parent.mkdir(parents=True, exist_ok=True)
        POINTS.write_text(json.dumps(saved, indent=1))
    return saved[video.stem]


def _run(video: Path, setup: str, points, config: GlassVideoConfig) -> dict:
    options = SETUPS[setup]
    try:
        demo, is_flip = read_video(video, config, points if options["claude"] else None)
    except VideoReadError as error:
        return {"video": video.stem, "setup": setup, "success": False, "refused": str(error)}
    report = run_robot(
        video,
        demo,
        is_flip,
        False,
        False,
        config,
        two_arms=options["two_arms"],
        suffix_extra=f"_setup_{setup}",
        choose_arms=options["choose_arms"],
    )
    return {
        "video": video.stem,
        "setup": setup,
        "success": report["success"],
        "failed_at": report.get("failed_at"),
        "arm": report.get("arm"),
        "model_chose": report.get("outcome_model", {}).get("chose"),
        "tower_error_cm": report.get("robot_tower_error_cm"),
        "refused": None,
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    config = GlassVideoConfig(**yaml.safe_load(CONFIG_PATH.read_text())["real_video"])
    saved = json.loads(POINTS.read_text()) if POINTS.exists() else {}
    results = []
    for name in VIDEOS:
        video = ROOT / "data/raw" / f"{name}.mov"
        points = _claude_points(video, saved)
        for setup in SETUPS:
            outcome = _run(video, setup, points, config)
            results.append(outcome)
            print(json.dumps(outcome), flush=True)
            OUTPUT.write_text(json.dumps(results, indent=1))
    for setup in SETUPS:
        runs = [r for r in results if r["setup"] == setup]
        print(f"Setup {setup}: {sum(r['success'] for r in runs)} of {len(runs)} built")


if __name__ == "__main__":
    main()
