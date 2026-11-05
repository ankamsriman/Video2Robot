"""Compare two ways of doing Task A on every real Task A video.

  A: Claude finds the glasses, and the logistic regression model picks the arms
     (each glass with the arm on its side, or one arm for all).
  B: Claude finds the glasses; each glass always uses the arm on its own side (no model).
  C: no Claude; each glass always uses the arm on its own side.

Claude is asked once per video, and its points are saved, so rerunning makes no API calls.
On the first 9 videos the model chose an arm per glass every time, so B would repeat A
there; B matters on new layouts. On cluttered tables only Claude can tell glasses from other
bright things, so C is pointless there.

PYTHONPATH=. .venv/bin/python scripts/compare_task2_setups.py
PYTHONPATH=. .venv/bin/python scripts/compare_task2_setups.py --videos task2_triangle_clutter_take1 \
    --setups A B --output docs/results/task2_clutter.json
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

from scripts.glass_from_video import (
    CONFIG_PATH,
    VideoReadError,
    glasses_from_model,
    glasses_near_end,
    read_video,
    run_robot,
)
from src.perception.glass_tracker import GlassVideoConfig
from src.simulation.scene import ROOT

VIDEOS = (
    "task2_triangle_right_take1",
    *(f"task2_triangle_take{n}" for n in range(2, 9)),
)
POINTS = ROOT / "data/processed/task2_setups/claude_points.json"
OUTPUT = ROOT / "docs/results/task2_setups.json"
SETUPS = {
    "A": {"claude": True, "two_arms": False, "choose_arms": True},
    "B": {"claude": True, "two_arms": True, "choose_arms": False},
    "C": {"claude": False, "two_arms": True, "choose_arms": False},
}


def _claude_points(video: Path, saved: dict) -> tuple[list, list]:
    """Claude's glasses near the start and near the end, asked once and then saved."""
    for key, at_end in ((video.stem, False), (f"{video.stem}@end", True)):
        if key not in saved:
            saved[key] = glasses_near_end(video) if at_end else glasses_from_model(video)
            POINTS.parent.mkdir(parents=True, exist_ok=True)
            POINTS.write_text(json.dumps(saved, indent=1))
    return saved[video.stem], saved[f"{video.stem}@end"]


def _run(video: Path, setup: str, points, config: GlassVideoConfig) -> dict:
    options = SETUPS[setup]
    try:
        start, end = points
        claude = options["claude"]
        demo, is_flip = read_video(
            video, config, start if claude else None, end if claude else None
        )
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", nargs="+", default=list(VIDEOS))
    parser.add_argument("--setups", nargs="+", default=["A", "C"], choices=list(SETUPS))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    config = GlassVideoConfig(**yaml.safe_load(CONFIG_PATH.read_text())["real_video"])
    saved = json.loads(POINTS.read_text()) if POINTS.exists() else {}
    results = []
    for name in args.videos:
        video = ROOT / "data/raw" / f"{name}.mov"
        points = _claude_points(video, saved)
        for setup in args.setups:
            outcome = _run(video, setup, points, config)
            results.append(outcome)
            print(json.dumps(outcome), flush=True)
            args.output.write_text(json.dumps(results, indent=1))
    for setup in args.setups:
        runs = [r for r in results if r["setup"] == setup]
        print(f"Setup {setup}: {sum(r['success'] for r in runs)} of {len(runs)} built")


if __name__ == "__main__":
    main()
