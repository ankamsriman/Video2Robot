"""Can a vision language model find the glasses on a cluttered table, and can the robot then
build what a typed instruction asks?

Each trial builds a random MuJoCo scene: three steel-coloured glasses and some other objects,
including a bright white box that looks like a glass top to a brightness rule. It renders a
picture from about where the user's phone stood, and asks the model for the glasses only.
The simulator knows the true positions, so every answer can be checked. The model's points
then become table positions, and the robot builds the triangle in the cluttered scene.

This uses rendered pictures, not real video. It needs VIDEO2ROBOT_CLAUDE_KEY in .env, and makes
two model requests per trial.

PYTHONPATH=. .venv/bin/python scripts/clutter_test.py --trials 5 --seed 4100
"""

import argparse
import dataclasses
import json
import logging

import cv2
import mujoco
import numpy as np
import yaml

from src.reasoning.fit_layout import arm_for_each_glass, first_goes_on_left, nearest_arm
from src.reasoning.instruction import InstructionError, choose_glasses
from src.reasoning.vlm_client import (
    AnthropicVision,
    VlmConfig,
    find_glass_tops,
    load_api_key,
    plan_from_instruction,
)
from src.simulation.glass_task import build_glass_triangle
from src.simulation.robot_env import RobotEnv, load_motion_config, motion_for_arm
from src.simulation.scene import ROOT, ItemConfig, build_model, load_scene_config
from src.verification.triangle import load_triangle_config

CONFIG = ROOT / "configs/cups_real.yaml"
OUTPUT = ROOT / "data/processed/clutter_test"
GLASSES = ("cup_a", "cup_b", "cup_c")
WIDTH, HEIGHT = 1280, 720
# Roughly where the user's phone stood: beyond the far edge, looking back and down.
CAMERA = {"lookat": (0.40, 0.0, 0.72), "distance": 1.35, "azimuth": 180.0, "elevation": -42.0}
GLASS_AREA = ((0.30, 0.48), (-0.28, 0.28))
TOWER = (0.46, 0.0)
GLASS_GAP = 0.13
CLEAR_OF_TOWER = 0.16
DISTRACTORS = (
    ("white_box", (0.06, 0.06, 0.09), (0.97, 0.97, 0.97, 1)),
    ("red_box", (0.07, 0.07, 0.05), (0.85, 0.15, 0.15, 1)),
    ("blue_bottle", (0.05, 0.05, 0.16), (0.15, 0.30, 0.85, 1)),
    ("yellow_block", (0.09, 0.05, 0.04), (0.95, 0.80, 0.15, 1)),
)
DISTRACTOR_AREA = ((0.20, 0.62), (-0.36, 0.36))
NEAR_MISS = 0.04
INSTRUCTION = "Ignore everything that isn't a drinking glass. Put the two glasses nearest to me at the bottom and the farthest one on top."
logger = logging.getLogger("clutter_test")


def _clear(point, others, gap) -> bool:
    return all(np.hypot(point[0] - o[0], point[1] - o[1]) >= gap for o in others)


def random_scene(rng: np.random.Generator) -> tuple[dict, dict]:
    """Glass and distractor positions, kept apart so the hand has room."""
    while True:
        glasses = [
            (rng.uniform(*GLASS_AREA[0]), rng.uniform(*GLASS_AREA[1])) for _ in GLASSES
        ]
        apart = all(_clear(g, glasses[:i], GLASS_GAP) for i, g in enumerate(glasses))
        if apart and _clear(TOWER, glasses, CLEAR_OF_TOWER):
            break
    placed = list(glasses) + [TOWER]
    others = {}
    for name, _, _ in DISTRACTORS:
        for _ in range(500):
            spot = (rng.uniform(*DISTRACTOR_AREA[0]), rng.uniform(*DISTRACTOR_AREA[1]))
            if _clear(spot, placed, 0.14):
                others[name] = spot
                placed.append(spot)
                break
    return dict(zip(GLASSES, glasses, strict=True)), others


def cluttered_config(base, distractors: dict):
    items = tuple(
        ItemConfig(name, size, distractors[name], rgba, mass=0.2)
        for name, size, rgba in DISTRACTORS
        if name in distractors
    )
    steel = tuple(
        dataclasses.replace(item, rgba=(0.62, 0.64, 0.68, 1)) for item in base.objects
    )
    return dataclasses.replace(base, objects=steel + items)


def _camera_pose() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Position, and right, up and forward directions, of MuJoCo's free camera."""
    azimuth, elevation = np.radians(CAMERA["azimuth"]), np.radians(CAMERA["elevation"])
    forward = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    position = np.array(CAMERA["lookat"]) - CAMERA["distance"] * forward
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return position, right, up, forward


def _focal(model: mujoco.MjModel) -> float:
    return (HEIGHT / 2) / np.tan(np.radians(model.vis.global_.fovy) / 2)


def to_pixel(point: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    position, right, up, forward = _camera_pose()
    offset = point - position
    depth = offset @ forward
    focal = _focal(model)
    return np.array(
        [WIDTH / 2 + focal * (offset @ right) / depth, HEIGHT / 2 - focal * (offset @ up) / depth]
    )


def from_pixel(pixel, height: float, model: mujoco.MjModel) -> np.ndarray:
    """The point at a given height whose picture is at `pixel`."""
    position, right, up, forward = _camera_pose()
    focal = _focal(model)
    ray = forward + ((pixel[0] - WIDTH / 2) / focal) * right - ((pixel[1] - HEIGHT / 2) / focal) * up
    distance = (height - position[2]) / ray[2]
    return position + distance * ray


def render(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = CAMERA["lookat"]
    camera.distance = CAMERA["distance"]
    camera.azimuth = CAMERA["azimuth"]
    camera.elevation = CAMERA["elevation"]
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        renderer.update_scene(data, camera)
        return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)


def bright_spots(picture: np.ndarray) -> int:
    """How many bright, colourless patches a brightness rule would see, as in the video reader."""
    hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    bright = ((hsv[..., 2] > 200) & (hsv[..., 1] < 45)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(bright)
    return int(sum(1 for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 150))


def run_trial(
    seed: int, model_client, vlm: VlmConfig, use_truth: bool = False, two_arms: bool = False
) -> dict:
    rng = np.random.default_rng(seed)
    glasses, distractors = random_scene(rng)
    base = load_scene_config(CONFIG)
    scene = cluttered_config(base, distractors)
    model = build_model(scene, {**glasses, **distractors})
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    picture = render(model, data)
    table = scene.table.top_height
    glass_height = base.objects[0].size[2]
    truth = {
        name: to_pixel(np.array([*spot, table + glass_height]), model)
        for name, spot in glasses.items()
    }
    _, encoded = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
    points = find_glass_tops(model_client, encoded.tobytes(), WIDTH, HEIGHT)
    found = {}
    for name, pixel in truth.items():
        if not points:
            break
        nearest = min(points, key=lambda p: np.hypot(p[0] - pixel[0], p[1] - pixel[1]))
        place = from_pixel(nearest, table + glass_height, model)[:2]
        if np.hypot(*(place - np.array(glasses[name]))) <= NEAR_MISS:
            found[name] = place
    false_points = 0
    for point in points:
        place = from_pixel(point, table + glass_height, model)[:2]
        if all(np.hypot(*(place - np.array(spot))) > NEAR_MISS for spot in glasses.values()):
            false_points += 1
    errors_cm = [
        round(float(np.hypot(*(found[n] - np.array(glasses[n])))) * 100, 1) for n in found
    ]
    record = {
        "seed": seed,
        "glasses_found": len(found),
        "model_points": len(points),
        "points_not_on_a_glass": false_points,
        "position_errors_cm": errors_cm,
        "bright_patches_a_brightness_rule_sees": bright_spots(picture),
    }
    marked = picture.copy()
    for pixel in truth.values():
        cv2.circle(marked, tuple(int(v) for v in pixel), 16, (0, 220, 0), 3)
    for point in points:
        cv2.drawMarker(marked, (int(point[0]), int(point[1])), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT / f"scene_{seed}.jpg"), marked)
    if len(found) < len(GLASSES) or false_points:
        record["robot"] = "not run: the glasses weren't found exactly"
        return record
    if use_truth:
        found = {name: np.array(spot) for name, spot in glasses.items()}
    starts = [found[name] for name in GLASSES]
    plan = plan_from_instruction(model_client, INSTRUCTION, [encoded.tobytes()], 3, False)
    order, _ = choose_glasses(starts, plan, video_has_flip=False)
    record["plan"] = plan.model_dump(mode="json")
    ordered = [GLASSES[i] for i in order]
    # The robot picks from the positions the model gave, not the simulator's true ones.
    layout_from_model = {name: tuple(found[name]) for name in GLASSES}
    run_model = build_model(scene, {**layout_from_model, **distractors})
    arm = nearest_arm([np.array(found[n]) for n in GLASSES], np.array(TOWER))
    arms = None
    first_on_left = arm == "right"
    if two_arms:
        arms = arm_for_each_glass([np.array(found[n]) for n in ordered])
        arm = arms[0]
        first_on_left = first_goes_on_left(arms)
    triangle = dataclasses.replace(
        load_triangle_config(CONFIG), center=TOWER, first_on_left=first_on_left
    )
    env = RobotEnv(run_model, motion_for_arm(load_motion_config(CONFIG), arm))
    result = build_glass_triangle(env, scene, triangle, tuple(ordered), arms)
    moved = [
        name
        for name, spot in distractors.items()
        if np.hypot(*(env.object_position(name)[:2] - np.array(spot))) > 0.02
    ]
    record.update(
        positions="true" if use_truth else "from the model",
        arm=arm if arms is None else list(arms),
        order=ordered,
        success=result.success,
        failures=list(result.failures),
        other_objects_disturbed=moved,
    )
    return record


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--truth", action="store_true", help="give the robot true positions")
    parser.add_argument("--two-arms", action="store_true")
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    vlm = VlmConfig(**yaml.safe_load(CONFIG.read_text())["vlm"])
    client = AnthropicVision(vlm, load_api_key(ROOT / ".env"))
    results = []
    for trial in range(args.trials):
        try:
            record = run_trial(args.seed + trial, client, vlm, args.truth, args.two_arms)
        except InstructionError as error:
            record = {"seed": args.seed + trial, "error": str(error)}
        results.append(record)
        print(json.dumps(record), flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"clutter{args.label}_seed_{args.seed}.json").write_text(json.dumps(results, indent=2))
    built = sum(bool(r.get("success")) for r in results)
    print(f"Result: robot built the triangle in {built} of {len(results)} cluttered scenes")


if __name__ == "__main__":
    main()
