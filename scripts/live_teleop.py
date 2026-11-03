"""Live control: your webcam watches your hand, and the simulated G1 copies it.

    PYTHONPATH=. .venv/bin/python scripts/live_teleop.py

Hold your hand up, facing the camera, until the window says "Following your hand". That
pose becomes the robot's starting pose. Then move, turn your wrist, and open or close your
thumb, index and middle finger. Keys: c resets the starting pose, q quits.

Without a camera, a recorded video can stand in for it:
    PYTHONPATH=. .venv/bin/python scripts/live_teleop.py \
        --source data/raw/task2_triangle_take6.mov --no-window --save out.mp4

Simulated G1 with a pinned pelvis. The robot copies your hand: nothing is learned.
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass

import cv2
import imageio
import mediapipe as mp
import mujoco
import numpy as np
import yaml
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from scipy.spatial.transform import Rotation, Slerp

from src.simulation.ik import palm_down_rotation
from src.simulation.mirror import mirror_point
from src.simulation.robot_env import RobotEnv, load_motion_config, motion_for_arm
from src.simulation.scene import ROOT, build_model, load_scene_config
from src.teleop.hand_retarget import (
    Calibration,
    TeleopConfig,
    finger_curls,
    finger_pose,
    hand_frame,
    image_scale,
    load_teleop_config,
    robot_rotation,
    smooth,
    wrist_target,
)

CONFIG_PATH = ROOT / "configs/cups_real.yaml"
HAND_MODEL = ROOT / "checkpoints/hand_landmarker.task"
GLASSES = ("cup_a", "cup_b", "cup_c")
# Live loops run as fast as they can; these bounds keep a slow frame from jolting physics.
MIN_STEP_SECONDS = 1 / 60
MAX_STEP_SECONDS = 0.1
SAVE_FPS = 15
BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
    (17, 18), (18, 19), (19, 20),
)  # fmt: skip
GREEN = (60, 200, 60)
YELLOW = (40, 200, 230)
RED = (60, 60, 230)
CAMERA_SIZE = (1280, 720)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HandReading:
    """One frame's hand, reduced to what the robot needs."""

    pixels: np.ndarray
    wrist: np.ndarray
    scale: float
    frame: np.ndarray
    curls: tuple[float, float, float]


def read_hand(
    landmarker: vision.HandLandmarker,
    image: np.ndarray,
    stamp_ms: int,
    config: TeleopConfig,
) -> HandReading | None:
    """The biggest hand in a BGR image, or None if no hand is seen."""
    height, width = image.shape[:2]
    rgb = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )
    found = landmarker.detect_for_video(rgb, stamp_ms)
    if not found.hand_landmarks:
        return None
    hands = [
        (
            np.array([[p.x * width, p.y * height] for p in image_hand]),
            np.array([[p.x, p.y, p.z] for p in world_hand]),
        )
        for image_hand, world_hand in zip(
            found.hand_landmarks, found.hand_world_landmarks, strict=True
        )
    ]
    pixels, world = max(hands, key=lambda hand: float(np.ptp(hand[0], axis=0).sum()))
    return HandReading(
        pixels=pixels,
        wrist=pixels[0] / (width, height),
        scale=image_scale(pixels, world),
        frame=hand_frame(world),
        curls=finger_curls(world, config),
    )


class HandFollower:
    """Turns hand readings into steady robot targets.

    Control starts once a hand is held still, so a hand walking into the picture can't
    set a bad starting pose. If the hand is gone for a while, it starts over from home.
    """

    def __init__(
        self, config: TeleopConfig, home: np.ndarray, neutral: np.ndarray
    ) -> None:
        self._config = config
        self._home = home
        self._neutral = neutral
        self.calibration: Calibration | None = None
        self._still_frames = 0
        self._anchor: np.ndarray | None = None
        self._lost_frames = 0
        self.position = home
        self.rotation = Rotation.from_matrix(neutral)
        self.curls = np.zeros(3)

    def recalibrate(self) -> None:
        self.calibration = None
        self._still_frames = 0
        self._anchor = None

    def _start_over(self) -> None:
        self.recalibrate()
        self.position = self._home
        self.rotation = Rotation.from_matrix(self._neutral)
        self.curls = np.zeros(3)
        logger.info("Hand gone: back to the starting pose")

    def _held_still(self, reading: HandReading) -> bool:
        """Count frames with the wrist staying near where it was first seen."""
        drift = (
            np.inf
            if self._anchor is None
            else float(np.max(np.abs(reading.wrist - self._anchor)))
        )
        if drift > self._config.calibration_still:
            self._anchor = reading.wrist
            self._still_frames = 0
        self._still_frames += 1
        return self._still_frames >= self._config.calibration_frames

    def update(self, reading: HandReading | None) -> None:
        if reading is None:
            self._lost_frames += 1
            if self._lost_frames == self._config.reset_after_lost:
                self._start_over()
            elif self.calibration is None:
                self._anchor = None
            return
        self._lost_frames = 0
        if self.calibration is None:
            if not self._held_still(reading):
                return
            self.calibration = Calibration(
                reading.wrist, reading.scale, reading.frame, self.position
            )
            logger.info("Calibrated: this hand pose is the robot's starting pose")
        weight = self._config.smoothing
        target = wrist_target(
            reading.wrist, reading.scale, self.calibration, self._config
        )
        self.position = smooth(self.position, target, weight)
        turned = Rotation.from_matrix(
            robot_rotation(
                self.calibration.frame, reading.frame, self._neutral, self._config
            )
        )
        between = Rotation.concatenate([self.rotation, turned])
        self.rotation = _blend(between, weight)
        self.curls = smooth(self.curls, np.array(reading.curls), weight)


def _blend(pair: Rotation, weight: float) -> Rotation:
    """Turn part of the way from the first rotation to the second."""
    return Slerp([0.0, 1.0], pair)(weight)


def _camera(view: dict) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = view["lookat"]
    camera.distance = view["distance"]
    camera.azimuth = view["azimuth"]
    camera.elevation = view["elevation"]
    return camera


def _draw_hand(image: np.ndarray, reading: HandReading | None) -> np.ndarray:
    """Your camera picture, mirrored like a mirror, with the tracked hand drawn on."""
    shown = cv2.flip(image, 1)
    if reading is None:
        return shown
    width = image.shape[1]
    points = [(int(width - x), int(y)) for x, y in reading.pixels]
    for start, end in BONES:
        cv2.line(shown, points[start], points[end], GREEN, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(shown, point, 4, YELLOW, -1, cv2.LINE_AA)
    return shown


def _status(follower: HandFollower, seen: bool, reachable: bool) -> tuple[str, tuple]:
    if follower.calibration is None:
        return "Hold your hand up and still, facing the camera...", YELLOW
    if not seen:
        return "Hand lost: the robot holds still", RED
    if not reachable:
        return "Out of the robot's reach", RED
    return "Following your hand", GREEN


def _compose(
    camera_image: np.ndarray, robot_image: np.ndarray, lines: list, colour
) -> np.ndarray:
    height = robot_image.shape[0]
    scale = height / camera_image.shape[0]
    # Video files need an even width, and webcams often give an odd one once scaled.
    width = 2 * round(camera_image.shape[1] * scale / 2)
    left = cv2.resize(camera_image, (width, height))
    right = cv2.cvtColor(robot_image, cv2.COLOR_RGB2BGR)
    out = np.hstack([left, right])
    cv2.rectangle(
        out, (0, 0), (out.shape[1], 34 + 24 * (len(lines) - 1)), (25, 25, 25), -1
    )
    for row, text in enumerate(lines):
        cv2.putText(
            out, text, (12, 24 + 24 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            colour if row == 0 else (230, 230, 230), 1 + (row == 0), cv2.LINE_AA,
        )  # fmt: skip
    return out


def _open_source(source: str) -> tuple[cv2.VideoCapture, bool]:
    is_camera = source.isdigit()
    capture = cv2.VideoCapture(int(source) if is_camera else source)
    if not capture.isOpened():
        hint = (
            " On a Mac, allow camera access for your terminal app in System Settings,"
            " Privacy & Security, Camera, then restart the terminal."
            if is_camera
            else ""
        )
        raise SystemExit(f"Could not open {source}.{hint}")
    if is_camera:
        # A smaller picture tracks the hand just as well, and keeps the loop quicker.
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
    return capture, is_camera


def _glass_layout(arm: str) -> dict[str, tuple[float, float]]:
    """The glasses start in front of the arm in use."""
    scene = load_scene_config(CONFIG_PATH)
    layout = {
        item.name: item.position for item in scene.objects if item.name in GLASSES
    }
    return {
        name: mirror_point(xy) if arm == "left" else xy for name, xy in layout.items()
    }


def _robot(arm: str) -> tuple[RobotEnv, dict]:
    model = build_model(load_scene_config(CONFIG_PATH), _glass_layout(arm))
    view = yaml.safe_load(CONFIG_PATH.read_text())["teleop_view"]
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, view["width"])
    model.vis.global_.offheight = max(model.vis.global_.offheight, view["height"])
    return RobotEnv(model, motion_for_arm(load_motion_config(CONFIG_PATH), arm)), view


def run(args: argparse.Namespace) -> dict:
    config = load_teleop_config(CONFIG_PATH)
    env, view = _robot(args.arm)
    follower = HandFollower(
        config,
        np.array(env.motion.home_position),
        palm_down_rotation(args.arm, env.motion.hand_pitch),
    )
    capture, is_camera = _open_source(args.source)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    writer = (
        imageio.get_writer(
            args.save, fps=SAVE_FPS if is_camera else fps, macro_block_size=1
        )
        if args.save
        else None
    )
    camera = _camera(view)
    counts = {"frames": 0, "hand_seen": 0, "following": 0, "out_of_reach": 0}
    started = last = time.monotonic()
    reachable = True
    with (
        vision.HandLandmarker.create_from_options(options) as landmarker,
        mujoco.Renderer(env.model, view["height"], view["width"]) as renderer,
    ):
        while True:
            ok, image = capture.read()
            if not ok or (args.seconds and time.monotonic() - started > args.seconds):
                break
            now = time.monotonic()
            step = (
                1 / fps
                if not is_camera
                else float(np.clip(now - last, MIN_STEP_SECONDS, MAX_STEP_SECONDS))
            )
            last = now
            stamp = (
                int((now - started) * 1000)
                if is_camera
                else int(counts["frames"] * 1000 / fps)
            )
            reading = read_hand(landmarker, image, stamp, config)
            follower.update(reading)
            counts["frames"] += 1
            counts["hand_seen"] += reading is not None
            # Follow a tracked hand; with no starting pose, go home; else hold still.
            if follower.calibration is None or reading is not None:
                reachable = env.command_hand(
                    follower.position,
                    follower.rotation.as_matrix(),
                    config.max_joint_speed * step,
                )
                env.command_fingers(finger_pose(tuple(follower.curls), config))
                following = follower.calibration is not None
                counts["following"] += following
                counts["out_of_reach"] += following and not reachable
            env.advance(step)
            renderer.update_scene(env.data, camera)
            text, colour = _status(follower, reading is not None, reachable)
            thumb, index, middle = follower.curls
            lines = [
                text,
                (
                    f"closed: thumb {thumb:.0%}  index {index:.0%}  middle {middle:.0%}"
                    "   |   c: new starting pose   q: quit"
                ),
            ]
            out = _compose(_draw_hand(image, reading), renderer.render(), lines, colour)
            if writer is not None:
                writer.append_data(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            if not args.no_window:
                cv2.imshow("Video2Robot live: your hand drives the G1", out)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    follower.recalibrate()
    capture.release()
    if writer is not None:
        writer.close()
    cv2.destroyAllWindows()
    seconds = time.monotonic() - started
    return {
        **counts,
        "loop_fps": round(counts["frames"] / seconds, 1),
        "glasses_moved_cm": {
            name: round(
                float(np.linalg.norm(env.object_position(name)[:2] - start)) * 100, 1
            )
            for name, start in _glass_layout(args.arm).items()
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="0", help="camera number, or a video file")
    parser.add_argument("--arm", choices=("right", "left"), default="right")
    parser.add_argument("--save", help="also save the window as an .mp4")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument(
        "--seconds", type=float, default=0.0, help="stop after this long"
    )
    summary = run(parser.parse_args())
    logger.info("%s", json.dumps(summary))


if __name__ == "__main__":
    main()
