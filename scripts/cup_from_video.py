"""Watch a video of someone building the cup triangle, and have the G1 build it too.

Film from the side, so that the two lower cups appear side by side and the top cup doesn't
hide them. Set each cup's colour in configs/cups.yaml first.

PYTHONPATH=. .venv/bin/python scripts/cup_from_video.py --video data/raw/cups_01.mp4
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.perception.observe_video import (
    load_color_specs,
    load_marker_config,
    load_tracking_config,
    observe_video,
)
from src.reasoning.demo_observer import ItemInfo, ObserverConfig
from src.reasoning.triangle_observer import observe_triangle
from src.simulation.cup_task import build_triangle
from src.simulation.recorder import video_recorder
from src.simulation.robot_env import RobotEnv, load_motion_config
from src.simulation.scene import ROOT, build_model, load_scene_config
from src.verification.triangle import load_triangle_config

CONFIG = ROOT / "configs/cups.yaml"
logger = logging.getLogger("cup_from_video")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()

    config = load_scene_config(CONFIG)
    items = tuple(ItemInfo(i.name, i.size, False) for i in config.objects)
    seen = observe_video(
        args.video,
        load_color_specs(CONFIG),
        items,
        load_marker_config(CONFIG),
        load_tracking_config(CONFIG),
    )
    settings = yaml.safe_load(CONFIG.read_text())["observer"]
    demo = observe_triangle(
        seen.tracks, seen.pixels, items, seen.fps, ObserverConfig(**settings)
    )
    logger.info("Placement order: %s", ", ".join(demo.order))
    logger.info("Build centre: (%.3f, %.3f)", *demo.center)

    output = ROOT / "data/processed" / args.video.stem
    model = build_model(config, demo.start_positions)
    with video_recorder(model, output / "rollout.mp4") as on_frame:
        env = RobotEnv(model, load_motion_config(CONFIG), on_frame=on_frame)
        triangle = demo.triangle_for(load_triangle_config(CONFIG))
        result = build_triangle(env, config, triangle, demo.order)
    report = {
        "success": result.success,
        "picked": list(result.picked),
        "failures": list(result.failures),
        "order": list(demo.order),
        "pelvis": "pinned",
    }
    (output / "result.json").write_text(json.dumps(report, indent=2))
    logger.info("Success: %s %s", result.success, list(result.failures))
    logger.info("Saved video to %s", output / "rollout.mp4")


if __name__ == "__main__":
    main()
