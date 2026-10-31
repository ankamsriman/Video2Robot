"""Check a test video and say what to fix, before recording the rest.

PYTHONPATH=. .venv/bin/python scripts/check_video.py --video data/raw/video_01.mp4
"""

import argparse
import logging
from pathlib import Path

import yaml

from src.perception.observe_video import load_color_specs, load_marker_config
from src.perception.video_check import Status, WorkArea, check_facts, gather_facts
from src.simulation.scene import CONFIG_PATH, load_scene_config

logger = logging.getLogger("check_video")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()

    config = load_scene_config()
    depths = {i.name: i.size[0] for i in (*config.objects, *config.containers)}
    facts = gather_facts(args.video, load_color_specs(), depths, load_marker_config())
    raw = yaml.safe_load(CONFIG_PATH.read_text())["work_area"]
    findings = check_facts(facts, WorkArea(tuple(raw["x"]), tuple(raw["y"])))
    for item in findings:
        logger.info("[%s] %s: %s", item.status.value, item.topic, item.message)
    problems = sum(item.status is Status.PROBLEM for item in findings)
    logger.info(
        "\n%s",
        "Looks good." if not problems else f"{problems} problem(s) to fix first.",
    )


if __name__ == "__main__":
    main()
