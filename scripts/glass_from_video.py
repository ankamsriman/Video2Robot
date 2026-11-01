"""Read a real triangle video of steel glasses, and have the G1 repeat it in simulation.

The table's rough corners are hand-picked once per video, in data/labels/<video>_table.json.
The video decides the order and the positions. The robot's grip is its own, and programmed.

PYTHONPATH=. .venv/bin/python scripts/glass_from_video.py --video data/raw/task2_triangle_right_take1.mov
"""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.perception.glass_tracker import (
    GlassVideoConfig,
    TriangleDemo,
    VideoReadError,
    find_tops,
    read_triangle,
    rest_spots,
    search_masks,
)
from src.perception.table_view import refine_corners, table_view
from src.simulation.glass_task import build_glass_triangle
from src.simulation.recorder import video_recorder
from src.simulation.robot_env import RobotEnv, load_motion_config
from src.simulation.scene import ROOT, SceneConfig, build_model, load_scene_config
from src.verification.triangle import load_triangle_config

CONFIG_PATH = ROOT / "configs/cups_real.yaml"
BACKGROUND_SAMPLES = 40
GLASS_NAMES = ("cup_a", "cup_b", "cup_c")
logger = logging.getLogger("glass_from_video")


def _background(capture: cv2.VideoCapture) -> np.ndarray:
    """Blend frames from across the video, which removes most of the moving person."""
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for index in np.linspace(0, count - 1, BACKGROUND_SAMPLES).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    if not frames:
        raise VideoReadError("The video has no readable frames")
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def read_video(video: Path, config: GlassVideoConfig) -> TriangleDemo:
    labels = ROOT / "data/labels" / f"{video.stem}_table.json"
    rough = np.array(json.loads(labels.read_text())["rough_corners"], dtype=float)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise VideoReadError(f"Could not open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    corners = refine_corners(_background(capture), rough)
    view = table_view(corners, config.table_width, config.table_depth, size)
    table, search = search_masks(size, corners, config.room_above_table)
    tops = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        tops.append(find_tops(frame, table, search, config))
    capture.release()
    return read_triangle(rest_spots(tops, fps, config), view, fps, config)


def _to_robot(place: np.ndarray, scene: SceneConfig, table_width: float) -> tuple[float, float]:
    """Table positions to simulator positions, with the robot at the middle of the edge."""
    return (
        float(place[0] + scene.table.near_edge_x),
        float(place[1] - table_width / 2),
    )


def run_robot(video: Path, demo: TriangleDemo, config: GlassVideoConfig) -> dict:
    scene = load_scene_config(CONFIG_PATH)
    motion = load_motion_config(CONFIG_PATH)
    starts = {
        name: _to_robot(place, scene, config.table_width)
        for name, place in zip(GLASS_NAMES, demo.starts, strict=True)
    }
    center = _to_robot(demo.tower_center, scene, config.table_width)
    triangle = dataclasses.replace(load_triangle_config(CONFIG_PATH), center=center)
    model = build_model(scene, starts)
    output = ROOT / "data/processed" / video.stem
    output.mkdir(parents=True, exist_ok=True)
    with video_recorder(model, output / "robot.mp4") as on_frame:
        env = RobotEnv(model, motion, on_frame=on_frame)
        result = build_glass_triangle(env, scene, triangle, GLASS_NAMES)
    placed = {name: env.object_position(name).tolist() for name in GLASS_NAMES}
    built = (np.array(placed["cup_a"])[:2] + np.array(placed["cup_b"])[:2]) / 2
    report = {
        "video": video.name,
        "success": result.success,
        "picked": list(result.picked),
        "failures": list(result.failures),
        "moved_at_seconds": [round(t, 1) for t in demo.moved_at],
        "starts_from_video": {k: [round(v, 3) for v in p] for k, p in starts.items()},
        "tower_from_video": [round(v, 3) for v in center],
        "lower_spacing_in_video_cm": round(demo.lower_spacing * 100, 1),
        "top_glass_offset_in_video_cm": round(demo.top_offset * 100, 1),
        "robot_tower_error_cm": round(float(np.linalg.norm(built - center)) * 100, 1),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(CONFIG_PATH.read_text())["real_video"]
    config = GlassVideoConfig(**raw)
    try:
        demo = read_video(args.video, config)
    except VideoReadError as error:
        logger.error("Couldn't read %s: %s", args.video.name, error)
        raise SystemExit(1) from error
    print(json.dumps(run_robot(args.video, demo, config), indent=2))


if __name__ == "__main__":
    main()
