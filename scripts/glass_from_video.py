"""Read a real video of steel glasses, and have the G1 repeat it in simulation.

Works for the triangle (three glasses) and the flipped tower (a fourth glass on top, with
its opening up). The kind of task comes from what the video shows.

The table's rough corners are hand-picked in data/labels: <video>_table.json for one video,
or current_setup_table.json for every new take from a camera that hasn't moved.
The video decides the order and the positions. The robot's grip and flip are programmed.

PYTHONPATH=. .venv/bin/python scripts/glass_from_video.py --video data/raw/task2_triangle_right_take1.mov
PYTHONPATH=. .venv/bin/python scripts/glass_from_video.py --video data/raw/task3_flip_right_take1.mov --fit-to-robot
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
    SPOTS_IN_FLIP,
    GlassVideoConfig,
    TriangleDemo,
    VideoReadError,
    find_tops,
    opening_faces_up,
    read_flip,
    read_triangle,
    rest_spots,
    search_masks,
)
from src.perception.table_view import refine_corners, table_view
from src.reasoning.fit_layout import Area, FittedLayout, fit_flip_layout
from src.simulation.cup_flip import build_flipped_tower, load_flip_config
from src.simulation.glass_task import build_glass_triangle
from src.simulation.recorder import video_recorder
from src.simulation.robot_env import RobotEnv, load_motion_config
from src.simulation.scene import ROOT, SceneConfig, build_model, load_scene_config
from src.verification.triangle import load_triangle_config

LABELS = ROOT / "data/labels"
CURRENT_SETUP = "current_setup_table.json"
CONFIG_PATH = ROOT / "configs/cups_real.yaml"
FLIP_CONFIG_PATH = ROOT / "configs/flip_real.yaml"
BACKGROUND_SAMPLES = 40
LAST_FRAMES = 30
GLASS_NAMES = ("cup_a", "cup_b", "cup_c")
FLIPPED_NAME = "cup_d"
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


def rough_corners(video: Path, labels: Path = LABELS) -> np.ndarray:
    """A video's own hand-picked corners, or else those of the current camera setup.

    New takes from a camera that hasn't moved share a table position, so they need no new
    corners. The edges are always fitted again from each video.
    """
    for name in (f"{video.stem}_table.json", CURRENT_SETUP):
        path = labels / name
        if path.exists():
            corners = json.loads(path.read_text())["rough_corners"]
            return np.array(corners, dtype=float)
    raise VideoReadError(f"No table corners for {video.name} in {labels}")


def read_video(video: Path, config: GlassVideoConfig) -> tuple[TriangleDemo, bool]:
    """Return what the video shows, and whether it shows a flipped fourth glass."""
    rough = rough_corners(video)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise VideoReadError(f"Could not open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    corners = refine_corners(_background(capture), rough)
    view = table_view(corners, config.table_width, config.table_depth, size)
    table, search = search_masks(size, corners, config.room_above_table)
    tops = []
    ending = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        tops.append(find_tops(frame, table, search, config))
        if len(tops) > count - LAST_FRAMES:
            ending.append(frame)
    capture.release()
    spots = rest_spots(tops, fps, view, config)
    if len(spots) != SPOTS_IN_FLIP:
        return read_triangle(spots, view, fps, config), False
    demo = read_flip(spots, view, fps, config)
    last_frame = np.median(np.stack(ending), axis=0).astype(np.uint8)
    if not opening_faces_up(last_frame, demo, view, config):
        raise VideoReadError(
            "Four glasses were seen, but the tower's top doesn't look like an opening"
        )
    return demo, True


def _to_robot(place: np.ndarray, scene: SceneConfig, table_width: float) -> np.ndarray:
    """Table positions to simulator positions, with the robot at the middle of the edge."""
    return np.array(
        [place[0] + scene.table.near_edge_x, place[1] - table_width / 2], dtype=float
    )


def _rounded(point: np.ndarray) -> list[float]:
    return [round(float(value), 3) for value in point]


def _fit(starts: list[np.ndarray], tower: np.ndarray) -> FittedLayout:
    raw = yaml.safe_load(FLIP_CONFIG_PATH.read_text())["flip_areas"]
    return fit_flip_layout(
        starts,
        tower,
        Area(tuple(raw["starts"]["x"]), tuple(raw["starts"]["y"])),
        Area(tuple(raw["flipped"]["x"]), tuple(raw["flipped"]["y"])),
        Area(tuple(raw["tower"]["x"]), tuple(raw["tower"]["y"])),
        raw["glass_gap"],
        raw["tower_gap"],
    )


def run_robot(
    video: Path, demo: TriangleDemo, is_flip: bool, fit: bool, config: GlassVideoConfig
) -> dict:
    """Place the glasses as the video shows, run the robot, and report the outcome."""
    path = FLIP_CONFIG_PATH if is_flip else CONFIG_PATH
    scene = load_scene_config(path)
    motion = load_motion_config(path)
    table_width = config.table_width
    starts = [_to_robot(place, scene, table_width) for place in demo.starts]
    tower = _to_robot(demo.tower_center, scene, table_width)
    report: dict = {
        "video": video.name,
        "task": "flipped tower" if is_flip else "triangle",
        "moved_at_seconds": [round(float(t), 1) for t in demo.moved_at],
        "starts_from_video": [_rounded(s) for s in starts],
        "tower_from_video": _rounded(tower),
        "lower_spacing_in_video_cm": round(demo.lower_spacing * 100, 1),
        "top_glass_offset_in_video_cm": round(demo.top_offset * 100, 1),
    }
    if fit:
        fitted = _fit(starts, tower)
        starts, tower = list(fitted.starts), fitted.tower
        report["moved_to_fit_robot"] = {
            "start_shifts_cm": [round(s * 100, 1) for s in fitted.start_shifts],
            "tower_shift_cm": round(fitted.tower_shift * 100, 1),
            "starts_used": [_rounded(s) for s in starts],
            "tower_used": _rounded(tower),
        }
    names = (*GLASS_NAMES, FLIPPED_NAME) if is_flip else GLASS_NAMES
    layout = {name: tuple(start) for name, start in zip(names, starts, strict=True)}
    triangle = dataclasses.replace(load_triangle_config(path), center=tuple(tower))
    model = build_model(scene, layout)
    suffix = "_fitted" if fit else ""
    output = ROOT / "data/processed" / video.stem
    output.mkdir(parents=True, exist_ok=True)
    with video_recorder(model, output / f"robot{suffix}.mp4") as on_frame:
        env = RobotEnv(model, motion, on_frame=on_frame)
        if is_flip:
            result = build_flipped_tower(
                env,
                scene,
                triangle,
                load_flip_config(path),
                GLASS_NAMES,
                FLIPPED_NAME,
                build_glass_triangle,
            )
        else:
            result = build_glass_triangle(env, scene, triangle, GLASS_NAMES)
    lower = [env.object_position(name)[:2] for name in GLASS_NAMES[:2]]
    built = (lower[0] + lower[1]) / 2
    report.update(
        success=result.success,
        failures=list(result.failures),
        robot_tower_error_cm=round(float(np.linalg.norm(built - tower)) * 100, 1),
    )
    (output / f"report{suffix}.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--fit-to-robot",
        action="store_true",
        help="for a flip, move positions to where the robot can flip, and report the shifts",
    )
    args = parser.parse_args()
    raw = yaml.safe_load(CONFIG_PATH.read_text())["real_video"]
    config = GlassVideoConfig(**raw)
    try:
        demo, is_flip = read_video(args.video, config)
    except VideoReadError as error:
        logger.error("Couldn't read %s: %s", args.video.name, error)
        raise SystemExit(1) from error
    report = run_robot(args.video, demo, is_flip, args.fit_to_robot, config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
