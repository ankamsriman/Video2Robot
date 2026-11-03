"""Set up live control to match a recorded video: same glasses, and the hand on the table.

The glasses are read from the video and placed where they really stood. The hand's distance
factor is fitted on where the glasses stood when the hand took them (see
`src/teleop/hand_on_table.py`). Used by `scripts/live_teleop.py --match-video`.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yaml
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from scripts.glass_from_video import (
    CONFIG_PATH,
    FLIP_CONFIG_PATH,
    FLIPPED_NAME,
    GLASS_NAMES,
    _to_robot,
    _video_view,
    read_video,
)
from src.perception.glass_tracker import GlassVideoConfig
from src.perception.table_view import TableView
from src.reasoning.fit_layout import nearest_arm
from src.simulation.robot_env import RobotEnv, load_motion_config, motion_for_arm
from src.simulation.scene import build_model, load_scene_config
from src.teleop.hand_on_table import (
    PalmRay,
    fit_depth_factor,
    palm_ray,
    palm_spot,
    table_to_robot,
)

# The hand is on a glass in the frames just before the reader sees it leave its spot.
GRAB_WINDOW_FRAMES = 8
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchedVideo:
    """A robot scene laid out like the video, and how to place the video's hand in it."""

    env: RobotEnv
    arm: str
    view: TableView
    factor: float
    fit_misses_cm: tuple[float, ...]
    starts: dict[str, np.ndarray]
    grab_frames: dict[int, str]
    near_edge_x: float
    table_width: float
    top_height: float
    is_flip: bool

    def robot_spot(self, pixels: np.ndarray, world: np.ndarray) -> np.ndarray:
        """Where on the simulated table a tracked palm is, in simulator coordinates."""
        spot = palm_spot(palm_ray(self.view, pixels, world), self.factor, self.view)
        return table_to_robot(spot, self.near_edge_x, self.table_width, self.top_height)


def _hands_per_frame(
    video: Path, model: Path
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Every frame's hands, as (pixels, metres) pairs."""
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            height, width = image.shape[:2]
            rgb = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            )
            found = landmarker.detect_for_video(rgb, int(len(frames) * 1000 / fps))
            frames.append(
                [
                    (
                        np.array([[p.x * width, p.y * height] for p in image_hand]),
                        np.array([[p.x, p.y, p.z] for p in world_hand]),
                    )
                    for image_hand, world_hand in zip(
                        found.hand_landmarks, found.hand_world_landmarks, strict=True
                    )
                ]
            )
    capture.release()
    return frames


def _rays_at_grab(
    hands: list, view: TableView, glass: np.ndarray, frame: int
) -> list[PalmRay]:
    """In each frame just before a glass moved, the palm nearest that glass."""
    rays = []
    for seen in hands[max(0, frame - GRAB_WINDOW_FRAMES) : frame + 1]:
        options = [palm_ray(view, pixels, world) for pixels, world in seen]
        if options:
            rays.append(
                min(
                    options,
                    key=lambda ray: float(
                        np.linalg.norm(palm_spot(ray, 1.0, view)[:2] - glass)
                    ),
                )
            )
    return rays


def match_video(
    video: Path, hand_model: Path, render_size: tuple[int, int]
) -> MatchedVideo:
    """Read the video's glasses and hand, and build a robot scene laid out the same way."""
    config = GlassVideoConfig(**yaml.safe_load(CONFIG_PATH.read_text())["real_video"])
    demo, is_flip = read_video(video, config)
    view = _video_view(video, config)
    path = FLIP_CONFIG_PATH if is_flip else CONFIG_PATH
    scene = load_scene_config(path)
    starts = [_to_robot(place, scene, config.table_width) for place in demo.starts]
    tower = _to_robot(demo.tower_center, scene, config.table_width)
    arm = nearest_arm(starts, tower)
    names = (*GLASS_NAMES, FLIPPED_NAME) if is_flip else GLASS_NAMES
    layout = {name: tuple(start) for name, start in zip(names, starts, strict=True)}
    model = build_model(scene, layout)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, render_size[0])
    model.vis.global_.offheight = max(model.vis.global_.offheight, render_size[1])
    env = RobotEnv(model, motion_for_arm(load_motion_config(path), arm))
    fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS)
    grab_frames = {
        round(float(when) * fps): name
        for when, name in zip(demo.moved_at, names, strict=True)
    }
    hands = _hands_per_frame(video, hand_model)
    rays = [
        _rays_at_grab(hands, view, glass, frame)
        for glass, frame in zip(demo.starts, grab_frames, strict=True)
    ]
    factor, misses = fit_depth_factor(rays, list(demo.starts), view)
    logger.info(
        "Glasses placed as in the video; hand distance factor %.2f, palm %s cm from each glass",
        factor,
        [round(m * 100, 1) for m in misses],
    )
    return MatchedVideo(
        env=env,
        arm=arm,
        view=view,
        factor=factor,
        fit_misses_cm=tuple(round(m * 100, 1) for m in misses),
        starts=dict(zip(names, (np.array(s) for s in starts), strict=True)),
        grab_frames=grab_frames,
        near_edge_x=scene.table.near_edge_x,
        table_width=config.table_width,
        top_height=scene.table.top_height,
        is_flip=is_flip,
    )
