"""The whole chain: demo video -> table positions -> plan -> G1 in physics -> retries.

Each stage saves its output under data/processed/<video name>/, so a stage can be rerun
alone, and the later stages can use tracking that was done elsewhere.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.keyframe_selector.hand_detector import save_hand_path, track_hand
from src.keyframe_selector.speed_analyzer import KeyframeConfig, select_keyframes
from src.perception.observe_video import (
    items_from_scene,
    load_color_specs,
    load_marker_config,
    load_tracking_config,
    observe_video,
)
from src.reasoning.demo_observer import DemoObservation, ObserverConfig, observe_demo
from src.schemas.plan import Plan
from src.simulation.primitives import final_relations
from src.simulation.recorder import video_recorder
from src.simulation.robot_env import RobotEnv, load_motion_config
from src.simulation.scene import CONFIG_PATH, ROOT, build_model, load_scene_config
from src.verification.closed_loop import SimulatedRobot, run_closed_loop

HAND_MODEL = ROOT / "checkpoints/hand_landmarker.task"
PROCESSED = ROOT / "data/processed"
DEFAULT_MAX_ATTEMPTS = 3
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunPaths:
    folder: Path

    @property
    def hand_path(self) -> Path:
        return self.folder / "hand_path.npz"

    @property
    def keyframes(self) -> Path:
        return self.folder / "keyframes.json"

    @property
    def tracks(self) -> Path:
        return self.folder / "tracks.npz"

    @property
    def observation(self) -> Path:
        return self.folder / "observation.json"

    @property
    def result(self) -> Path:
        return self.folder / "result.json"

    @property
    def rollout(self) -> Path:
        return self.folder / "rollout.mp4"


def paths_for(video: Path, root: Path = PROCESSED) -> RunPaths:
    folder = root / video.stem
    folder.mkdir(parents=True, exist_ok=True)
    return RunPaths(folder)


def run_keyframes(video: Path, paths: RunPaths) -> tuple[float, ...]:
    """Find key moments from the hand. A video with no visible hand gives none."""
    settings = yaml.safe_load(CONFIG_PATH.read_text())["keyframes"]
    hand = track_hand(video, HAND_MODEL)
    save_hand_path(hand, paths.hand_path)
    try:
        result = select_keyframes(hand.centers, hand.fps, KeyframeConfig(**settings))
    except ValueError as error:
        logger.warning("Skipping key moments: %s", error)
        paths.keyframes.write_text(json.dumps({"times": []}))
        return ()
    paths.keyframes.write_text(json.dumps({"times": list(result.times)}, indent=2))
    return result.times


def run_perception(video: Path, paths: RunPaths) -> None:
    """Follow every coloured item, and save its table position in each frame."""
    config = load_scene_config()
    specs = load_color_specs()
    colored = {spec.name for spec in specs}
    items = tuple(i for i in items_from_scene(config) if i.name in colored)
    seen = observe_video(
        video, specs, items, load_marker_config(), load_tracking_config()
    )
    np.savez_compressed(paths.tracks, fps=seen.fps, **seen.tracks)


def _with_fixed_items(tracks: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Items with no trackable colour, like a white tray, stay at their set positions."""
    frames = len(next(iter(tracks.values())))
    full = dict(tracks)
    for item in (*load_scene_config().objects, *load_scene_config().containers):
        if item.name not in full:
            logger.warning("'%s' has no colour, so its set position is used", item.name)
            full[item.name] = np.tile(item.position, (frames, 1)).astype(float)
    return full


def run_planning(paths: RunPaths) -> DemoObservation:
    """Work out the plan and starting positions from the saved tracks."""
    with np.load(paths.tracks) as saved:
        fps = float(saved["fps"])
        tracks = {name: saved[name] for name in saved.files if name != "fps"}
    settings = yaml.safe_load(CONFIG_PATH.read_text())["observer"]
    demo = observe_demo(
        _with_fixed_items(tracks),
        items_from_scene(load_scene_config()),
        fps,
        ObserverConfig(**settings),
    )
    paths.observation.write_text(
        json.dumps(
            {
                "plan": demo.plan.model_dump(mode="json"),
                "start_positions": dict(demo.start_positions),
                "final_positions": dict(demo.final_positions),
            },
            indent=2,
        )
    )
    return demo


def run_simulation(paths: RunPaths, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> dict:
    """Have the G1 carry out the plan, retrying failed steps, and record the result."""
    saved = json.loads(paths.observation.read_text())
    plan = Plan.model_validate(saved["plan"])
    starts = {name: tuple(xy) for name, xy in saved["start_positions"].items()}
    config = load_scene_config()
    motion = load_motion_config()
    model = build_model(config, starts)
    with video_recorder(model, paths.rollout) as on_frame:
        env = RobotEnv(model, motion, on_frame=on_frame)
        outcome = run_closed_loop(
            SimulatedRobot(env, config), plan, motion, max_attempts
        )
    final = final_relations(env, config, plan)
    result = {
        "success": outcome.success,
        "attempts": outcome.attempts,
        "total_attempts": outcome.total_attempts,
        "failures": list(outcome.failures),
        "steps": len(plan.steps),
        "final_relations": [relation.value for relation in final],
        "pelvis": "pinned",
    }
    paths.result.write_text(json.dumps(result, indent=2))
    return result
