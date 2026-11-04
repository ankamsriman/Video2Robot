"""Read a real video of steel glasses, and have the G1 repeat it in simulation.

Works for the triangle (three glasses) and the flipped tower (a fourth glass on top, with
its opening up). The kind of task comes from what the video shows.

The table's rough corners are hand-picked in data/labels: <video>_table.json for one video,
or current_setup_table.json for every new take from a camera that hasn't moved.
The video decides the order and the positions. The robot's grip and flip are programmed.

For a triangle, the robot stays mid-table and uses the arm nearest the glasses and tower.
For a flip, the video's layout also decides which arm works and where the robot stands along
the table, so no glass has to move. --fit-to-robot instead keeps the robot in the middle, using
its right arm, and moves the glasses to suit it. --stay-in-middle keeps the robot there and
moves nothing.

PYTHONPATH=. .venv/bin/python scripts/glass_from_video.py --video data/raw/task2_triangle_right_take1.mov
PYTHONPATH=. .venv/bin/python scripts/glass_from_video.py --video data/raw/task3_flip_right_take1.mov
"""

import argparse
import dataclasses
import itertools
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.keyframe_selector.hand_detector import track_both_hands
from src.perception.glass_tracker import (
    GLASSES_IN_TRIANGLE,
    SPOTS_IN_FLIP,
    GlassVideoConfig,
    Spot,
    TriangleDemo,
    VideoReadError,
    find_tops,
    opening_faces_up,
    read_flip,
    read_triangle,
    rest_spots,
    search_masks,
)
from src.perception.hand_trajectory import (
    CarryLimits,
    carry_window,
    follow_moving_hand,
    is_believable,
    profile_from_pixels,
)
from src.perception.sam_reader import spots_from_tracks
from src.perception.table_view import TableView, refine_corners, table_view
from src.perception.vlm_rescue import (
    arrival_frame,
    departure_frame,
    departure_from_order,
    drop_clutter,
    drop_misread_ends,
    later_arrivals,
    missing_points,
    patch_change,
    rescued_spot,
    seen_from_start,
    trust_model_departures,
    unseen_ends,
    unused_glasses,
)
from src.reasoning.fit_layout import (
    Area,
    FittedLayout,
    arm_for_each_glass,
    first_goes_on_left,
    fit_flip_layout,
    nearest_arm,
    plan_stand,
)
from src.reasoning.instruction import InstructionError, choose_glasses
from src.reasoning.outcome_model import MODES, features, prefer_two_arms, table_cm, train
from src.reasoning.vlm_client import (
    AnthropicVision,
    VlmConfig,
    find_glass_tops,
    load_api_key,
    plan_from_instruction,
    video_frames,
)
from src.simulation.cup_flip import build_flipped_tower, flip_for_arm, load_flip_config
from src.simulation.glass_task import CopiedCarries, build_glass_triangle
from src.simulation.recorder import video_recorder
from src.simulation.robot_env import RobotEnv, load_motion_config, motion_for_arm
from src.simulation.scene import ROOT, SceneConfig, build_model, load_scene_config
from src.verification.steps import first_failure, step_summary
from src.verification.triangle import load_triangle_config

LABELS = ROOT / "data/labels"
CURRENT_SETUP = "current_setup_table.json"
CONFIG_PATH = ROOT / "configs/cups_real.yaml"
HAND_MODEL = ROOT / "checkpoints/hand_landmarker.task"
FLIP_CONFIG_PATH = ROOT / "configs/flip_real.yaml"
BACKGROUND_SAMPLES = 40
LAST_FRAMES = 30
GLASS_NAMES = ("cup_a", "cup_b", "cup_c")
FLIPPED_NAME = "cup_d"
# If the model gives no glasses for the last picture, try these earlier ones.
END_TRIES_SECONDS = (0.5, 1.5, 3.0)
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


def glasses_near_end(video: Path) -> list[tuple[float, float]]:
    """Claude's glasses near the end, trying slightly earlier pictures if one gets no answer.

    The last moments can show the person filling the picture as they step away. If no
    picture gets an answer, the end check is skipped and the reader's ends are used.
    """
    for seconds in END_TRIES_SECONDS:
        try:
            return glasses_from_model(video, at_end=True, before_end=seconds)
        except InstructionError as error:
            logger.warning("No glasses from the model %.1f s before the end: %s", seconds, error)
    return []


def glasses_from_model(
    video: Path, at_end: bool = False, before_end: float | None = None
) -> list[tuple[float, float]]:
    """Ask a vision language model where every glass top is, in pixels.

    It looks early in the video, or with `at_end` near its end, where the glasses finished.
    """
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    vlm = VlmConfig(**raw["vlm"])
    rescue = raw["vlm_rescue"]
    capture = cv2.VideoCapture(str(video))
    if at_end:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        seconds = rescue["end_seconds"] if before_end is None else before_end
        before = round(seconds * capture.get(cv2.CAP_PROP_FPS))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(count - 1 - before, 0))
    else:
        capture.set(cv2.CAP_PROP_POS_MSEC, rescue["frame_seconds"] * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise VideoReadError(f"Couldn't read an early frame of {video.name}")
    scale = vlm.image_width / frame.shape[1]
    small = cv2.resize(frame, (vlm.image_width, int(frame.shape[0] * scale)))
    model = AnthropicVision(vlm, load_api_key(ROOT / ".env"))
    picture = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    return [
        (x / scale, y / scale)
        for x, y in find_glass_tops(model, picture, small.shape[1], small.shape[0])
    ]


def read_video(
    video: Path,
    config: GlassVideoConfig,
    model_points: list[tuple[float, float]] | None = None,
    end_points: list[tuple[float, float]] | None = None,
    sam_tracks: dict | None = None,
) -> tuple[TriangleDemo, bool]:
    """Return what the video shows, and whether it shows a flipped fourth glass.

    `model_points` are glass tops a vision model found early in the video. Any the reader
    missed are added as starting spots.
    """
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
    points = model_points or []
    finals = end_points or []
    rescue = yaml.safe_load(CONFIG_PATH.read_text())["vlm_rescue"] if points or finals else {}
    end_patches: list[list[np.ndarray]] = [[] for _ in finals]
    references: list = []
    changes: list[list[float]] = [[] for _ in points]
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        tops.append(find_tops(frame, table, search, config))
        if finals:
            gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for index, point in enumerate(finals):
                end_patches[index].append(_patch(gray_now, point, rescue["patch_half_pixels"]))
        if points:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            half = rescue["patch_half_pixels"]
            if len(tops) == rescue["reference_frame"] + 1:
                references = [
                    gray[int(y) - half : int(y) + half, int(x) - half : int(x) + half].astype(float)
                    for x, y in points
                ]
            for index, point in enumerate(points):
                if references:
                    changes[index].append(patch_change(gray, point, half, references[index]))
        if len(tops) > count - LAST_FRAMES:
            ending.append(frame)
    capture.release()
    spots = rest_spots(tops, fps, view, config)
    unused: list = []
    if sam_tracks is not None:
        # SAM 2 followed each glass by its outline, so its tracks replace the bright-top
        # spots and the rules that patch them; every later check still applies.
        settings = yaml.safe_load(CONFIG_PATH.read_text())["sam_reader"]
        near = (rescue or yaml.safe_load(CONFIG_PATH.read_text())["vlm_rescue"])["near_pixels"]
        spots, unused = spots_from_tracks(
            sam_tracks["tracks"], points, finals, fps,
            settings["moved_pixels"], settings["stay_seconds"], near,
        )  # fmt: skip
        points, finals = [], []
    if points and finals:
        spots, points, changes, finals, end_patches, unused = _set_aside_unused(
            spots, points, changes, finals, end_patches, fps, rescue, config
        )

    unknown: list = []
    if points:
        spots, unknown = _add_rescued(spots, points, changes, fps, rescue)
    if points and finals:
        departures = [
            None if left is None else left + rescue["reference_frame"]
            for left in (
                departure_frame(
                    np.array(series), rescue["change_threshold"], fps, rescue["settle_seconds"]
                )
                for series in changes
            )
        ]
        spots, gone = trust_model_departures(
            spots, points, finals, departures, len(tops) - 1, fps,
            config.run_gap_seconds, rescue["near_pixels"],
        )  # fmt: skip
        unknown = [*unknown, *gone]
    if finals:
        def tower_top(spot) -> bool:
            """A spot centred above two glasses the model sees at the end: a stacked glass.

            In a flip the triangle's top glass is covered at the end, so the model can't
            see it; this keeps it while other mid-video spots are judged as usual.
            """
            lower = [view.position(point, config.glass_height) for point in finals]
            raised = view.position((spot.x, spot.y), 2 * config.glass_height)
            return any(
                np.linalg.norm(raised - (a + b) / 2) <= config.max_top_offset
                for a, b in itertools.combinations(lower, 2)
            )

        spots = _match_ends(
            spots, finals, end_patches, fps, rescue, config, len(unknown), len(points),
            tower_top,
        )  # fmt: skip
    spots = _add_unseen_departure(spots, unknown, fps, rescue) if unknown else spots
    untouched = tuple(view.position(point, config.glass_height) for point in unused)
    if len(spots) != SPOTS_IN_FLIP:
        demo = _read_triangle_retimed(spots, finals, end_patches, view, fps, rescue, config)
        return dataclasses.replace(demo, untouched=untouched), False
    demo = dataclasses.replace(read_flip(spots, view, fps, config), untouched=untouched)
    last_frame = np.median(np.stack(ending), axis=0).astype(np.uint8)
    if not opening_faces_up(last_frame, demo, view, config):
        raise VideoReadError(
            "Four glasses were seen, but the tower's top doesn't look like an opening"
        )
    return demo, True


def _video_view(video: Path, config: GlassVideoConfig) -> TableView:
    capture = cv2.VideoCapture(str(video))
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    corners = refine_corners(_background(capture), rough_corners(video))
    capture.release()
    return table_view(corners, config.table_width, config.table_depth, size)


def copied_carries(
    video: Path, demo: TriangleDemo, config: GlassVideoConfig
) -> tuple[CopiedCarries, list[dict]]:
    """Read the hand's path for each of the three carries, from when its glass left."""
    raw = yaml.safe_load(CONFIG_PATH.read_text())["hand_path"]
    limits = CarryLimits(**{key: raw[key] for key in CarryLimits.__dataclass_fields__})
    hands, fps = track_both_hands(video, HAND_MODEL)
    view = _video_view(video, config)
    height = config.glass_height
    profiles = []
    details = []
    for index in range(3):
        start, end = demo.starts[index], demo.ends[index]
        lift = 1.5 * height if index == 2 else 0.5 * height
        start_pixel = view.pixel(start, 0.5 * height)
        end_pixel = view.pixel(end, lift)
        begin = max(int((demo.moved_at[index] - raw["lead_seconds"]) * fps), 0)
        stop = len(hands)
        if index + 1 < len(demo.moved_at):
            stop = int((demo.moved_at[index + 1] + 0.0) * fps)
        path = follow_moving_hand(hands[begin:stop], start_pixel, raw["max_jump_pixels"])
        first, last = carry_window(path, start_pixel, end_pixel, raw["settle_pixels"])
        profile = profile_from_pixels(path[first : last + 1], fps, view, start, end)
        believable = is_believable(profile, raw["believable_start"], raw["believable_top"])
        profiles.append(profile if believable else None)
        details.append(
            {
                "copied": believable,
                "seconds": round(profile.seconds, 2),
                "highest_cm": round(float(profile.height.max()) * 100, 1),
                "from_second": round((begin + first) / fps, 2),
            }
        )
    return CopiedCarries(tuple(profiles), limits), details


def follow_instruction(
    video: Path, demo: TriangleDemo, is_flip: bool, instruction: str
) -> tuple[TriangleDemo, bool, np.ndarray | None, dict]:
    """Let a vision language model choose the task and order. Positions stay as measured.

    Returns the reordered demonstration, whether to flip, any glass left out of a triangle,
    and what to report.
    """
    raw = yaml.safe_load(CONFIG_PATH.read_text())["vlm"]
    vlm = VlmConfig(**raw)
    model = AnthropicVision(vlm, load_api_key(ROOT / ".env"))
    images = video_frames(video, vlm.frames, vlm.image_width)
    plan = plan_from_instruction(model, instruction, images, len(demo.starts), is_flip)
    order, flip = choose_glasses(demo.starts, plan, is_flip)
    left_out = demo.starts[3] if is_flip and not flip else None
    chosen = dataclasses.replace(
        demo,
        starts=tuple(demo.starts[i] for i in order),
        moved_at=tuple(demo.moved_at[i] for i in order),
    )
    report = {
        "instruction": instruction,
        "model": vlm.model,
        "plan": plan.model_dump(mode="json"),
        "glasses_in_video_order_used": [i + 1 for i in order],
        "glass_left_out": left_out is not None,
    }
    return chosen, flip, left_out, report


def _set_aside_unused(spots, points, changes, finals, end_patches, fps, rescue, config):
    """Take glasses that never moved out of the task, as the model and the reader agree."""
    departures = [
        departure_frame(np.array(series), rescue["change_threshold"], fps, rescue["settle_seconds"])
        for series in changes
    ]
    last = len(end_patches[0]) - 1
    near = rescue["near_pixels"]
    unused = unused_glasses(
        points, finals, spots, departures, last, fps, config.run_gap_seconds, near
    )
    if not unused:
        return spots, points, changes, finals, end_patches, unused

    def is_unused(point) -> bool:
        return any(np.hypot(point[0] - u[0], point[1] - u[1]) <= near for u in unused)

    logger.warning("%d glass(es) never moved, so they take no part: %s", len(unused), unused)
    keep_start = [i for i, point in enumerate(points) if not is_unused(point)]
    keep_end = [i for i, point in enumerate(finals) if not is_unused(point)]
    return (
        tuple(spot for spot in spots if not is_unused((spot.x, spot.y))),
        [points[i] for i in keep_start],
        [changes[i] for i in keep_start],
        [finals[i] for i in keep_end],
        [end_patches[i] for i in keep_end],
        unused,
    )


def _patch(gray: np.ndarray, point: tuple[float, float], half: int) -> np.ndarray:
    x, y = int(point[0]), int(point[1])
    return gray[max(y - half, 0) : y + half, max(x - half, 0) : x + half].astype(float)


def _match_ends(
    spots, finals, patches, fps, rescue, config, unknown: int = 0, glasses: int = 0,
    tower_top=None,
) -> tuple:
    """Check the glasses at the end against the model's view of the last frames.

    A spot still there at the end that matches no glass the model saw is a misread, and is
    dropped. For a triangle, a glass the model saw that the reader missed, such as a dim
    top, is added; when it arrived is measured from a patch of the picture at its spot.
    """
    last = len(patches[0]) - 1
    gap, near = config.run_gap_seconds, rescue["near_pixels"]
    spots = drop_misread_ends(spots, finals, last, fps, gap, near, exempt=tower_top)
    starts = [spot for spot in spots if spot.first / fps <= rescue["early_seconds"]]
    if len(starts) + unknown != GLASSES_IN_TRIANGLE:
        return spots
    reference = max(last - round(rescue["end_seconds"] * fps), 0)
    added = []
    for point in unseen_ends(spots, finals, last, fps, gap, near):
        arrived = _patch_arrival(patches[finals.index(point)], reference, fps, rescue)
        if arrived is None:
            logger.warning("A glass the model saw at the end at %s never seemed to arrive", point)
            continue
        added.append(Spot(float(point[0]), float(point[1]), arrived, last, last - arrived + 1))
    if added:
        logger.warning("Added %d end glass(es) the reader missed", len(added))
    return tuple(sorted((*spots, *added), key=lambda spot: spot.first))


def _patch_arrival(series, reference: int, fps: float, rescue: dict) -> int | None:
    """When the picture at one end point began to match how it looks near the end."""
    change = np.array(
        [float(np.mean(np.abs(patch - series[reference]))) for patch in series[: reference + 1]]
    )
    return arrival_frame(change, rescue["change_threshold"], fps, rescue["settle_seconds"])


def _read_triangle_retimed(spots, finals, patches, view, fps, rescue, config) -> TriangleDemo:
    """Read a triangle; if a landing seems to come too early, time it from the pictures.

    Only used when the plain reading is impossible, so videos that read already can't change.
    """
    try:
        return read_triangle(spots, view, fps, config)
    except VideoReadError as error:
        if not finals or "arrive before" not in str(error):
            raise
    last = len(patches[0]) - 1
    reference = max(last - round(rescue["end_seconds"] * fps), 0)
    arrivals = [_patch_arrival(series, reference, fps, rescue) for series in patches]
    early = config.run_gap_seconds * fps
    landed = [spot for spot in spots if spot.first > early]
    moved = dict(zip(landed, later_arrivals(landed, finals, arrivals, rescue["near_pixels"])))
    retimed = tuple(moved.get(spot, spot) for spot in spots)
    logger.warning("Timed %d landing(s) from the pictures at the end points",
                   sum(moved[spot] != spot for spot in landed))
    return read_triangle(retimed, view, fps, config)


def _add_rescued(spots, points, changes, fps, rescue) -> tuple[tuple, list]:
    """Drop bright clutter, and add a starting spot for each glass the reader missed.

    """
    spots = drop_clutter(
        spots, points, fps, rescue["early_seconds"], rescue["near_pixels"]
    )
    spots = seen_from_start(
        points, spots, fps, rescue["early_seconds"], rescue["near_pixels"]
    )
    missing = missing_points(
        points, spots, fps, rescue["early_seconds"], rescue["near_pixels"]
    )
    added = []
    unknown = []
    for point in missing:
        index = points.index(point)
        series = np.array(changes[index])
        left = departure_frame(
            series, rescue["change_threshold"], fps, rescue["settle_seconds"]
        )
        if left is None:
            logger.warning("A glass the model found at %s never seemed to leave", point)
            unknown.append(point)
            continue
        added.append(rescued_spot(point, left + rescue["reference_frame"]))
    logger.warning("Added %d starting glass(es) the reader missed", len(added))
    return tuple(sorted((*spots, *added), key=lambda spot: spot.first)), unknown


def _add_unseen_departure(spots, unknown, fps, rescue) -> tuple:
    """Give the one starting glass whose leaving wasn't seen a departure, from the order."""
    if len(unknown) != 1:
        return spots
    early = rescue["early_seconds"] * fps
    starts = [spot.last for spot in spots if spot.first <= early]
    landings = [spot.first for spot in spots if spot.first > early]
    left = departure_from_order(starts, landings)
    if left is None:
        return spots
    logger.warning("Worked out when the glass at %s left, from the order of moves", unknown[0])
    return tuple(sorted((*spots, rescued_spot(unknown[0], left)), key=lambda spot: spot.first))


def _model_prefers_two_arms(starts, tower, scene: SceneConfig, report: dict) -> bool:
    """Train the outcome model on saved runs, and ask which way it expects to work."""
    settings = yaml.safe_load(CONFIG_PATH.read_text())["outcome_model"]
    seeds = settings["training_seeds"]
    rows = []
    for seed in seeds:
        for mode in MODES:
            runs = json.loads((ROOT / f"docs/results/triangle_{mode}_whole_seed_{seed}.json").read_text())
            rows += [(features(r["starts"], r["tower"], mode), r["success"]) for r in runs]
    model = train(rows)
    edge = scene.table.near_edge_x
    cm_starts = [table_cm(start, edge) for start in starts]
    cm_tower = table_cm(tower, edge)
    chances = {mode: round(model.chance(cm_starts, cm_tower, mode), 2) for mode in MODES}
    two = prefer_two_arms(chances, settings.get("two_arms_margin", 0.0))
    report["outcome_model"] = {"predicted_chance": chances, "chose": "two" if two else "nearest"}
    return two


def _to_robot(place: np.ndarray, scene: SceneConfig, table_width: float) -> np.ndarray:
    """Table positions to simulator positions, with the robot at the middle of the edge."""
    return np.array(
        [place[0] + scene.table.near_edge_x, place[1] - table_width / 2], dtype=float
    )


def _flip_tower_reach() -> float:
    """How far to the working arm's side the tower stands, from the middle of the tested area."""
    raw = yaml.safe_load(FLIP_CONFIG_PATH.read_text())["flip_areas"]
    return abs(sum(raw["tower"]["y"]) / 2)


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
    video: Path,
    demo: TriangleDemo,
    is_flip: bool,
    fit: bool,
    stay_in_middle: bool,
    config: GlassVideoConfig,
    two_arms: bool = False,
    hand_path: bool = False,
    left_out: np.ndarray | None = None,
    suffix_extra: str = "",
    notes: dict | None = None,
    choose_arms: bool = False,
    copy_side_order: bool = False,
) -> dict:
    """Place the glasses as the video shows, run the robot, and report the outcome.

    `left_out` is a glass that stays on the table untouched, as when an instruction asks
    for a triangle from a video with four glasses.
    """
    if left_out is None and demo.untouched and not is_flip:
        left_out = demo.untouched[0]
        notes = {**(notes or {}), "untouched_glass_from_video": _rounded(left_out)}
    path = FLIP_CONFIG_PATH if is_flip or left_out is not None else CONFIG_PATH
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
        "top_glass_seen_in_video": demo.top_seen,
        "top_glass_offset_in_video_cm": (
            round(demo.top_offset * 100, 1) if demo.top_seen else None
        ),
        **(notes or {}),
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
    arm = motion.arm
    arms = None
    first_on_left = None
    if not is_flip:
        arm = nearest_arm(starts, tower)
        report["arm"] = arm
        if choose_arms:
            two_arms = _model_prefers_two_arms(starts, tower, scene, report)
        if two_arms:
            arms = arm_for_each_glass(starts)
            arm = arms[0]
            first_on_left = first_goes_on_left(arms)
            report["arm"] = list(arms)
    if is_flip and not (fit or stay_in_middle):
        stand = plan_stand(starts, tower, _flip_tower_reach(), table_width / 2)
        arm = stand.arm
        starts = [start - [0.0, stand.robot_y] for start in starts]
        tower = tower - [0.0, stand.robot_y]
        scene = dataclasses.replace(
            scene, table=dataclasses.replace(scene.table, center_y=-stand.robot_y)
        )
        report["robot_stands"] = {
            "arm": arm,
            "cm_along_table_from_middle_to_its_left": round(stand.robot_y * 100, 1),
            "glasses_moved": False,
        }
    names = (*GLASS_NAMES, FLIPPED_NAME) if is_flip else GLASS_NAMES
    layout = {name: tuple(start) for name, start in zip(names, starts, strict=True)}
    if left_out is not None:
        layout[FLIPPED_NAME] = tuple(_to_robot(left_out, scene, table_width))
    video_first_left = bool(demo.ends[0][1] > demo.ends[1][1])
    report["first_lower_glass_side_from_video"] = "left" if video_first_left else "right"
    if copy_side_order and not is_flip:
        # Copy which side the person put their first lower glass on. A larger "left" is
        # further to the robot's left, the same in table and simulator positions. Off by
        # default: on take 4 and hard take 1 it made the hand reach past a placed glass.
        first_on_left = video_first_left
    triangle = dataclasses.replace(
        load_triangle_config(path),
        center=tuple(tower),
        first_on_left=arm == "right" if first_on_left is None else first_on_left,
    )
    model = build_model(scene, layout)
    suffix = "_fitted" if fit else "_middle" if stay_in_middle else "_two_arms" if two_arms else ""
    suffix = f"{suffix}_hand_path" if hand_path else suffix
    suffix = f"{suffix}{suffix_extra}"
    carries = None
    if hand_path:
        if is_flip:
            raise SystemExit("--hand-path only works for a triangle so far")
        carries, report["hand_carries"] = copied_carries(video, demo, config)
    output = ROOT / "data/processed" / video.stem
    output.mkdir(parents=True, exist_ok=True)
    with video_recorder(model, output / f"robot{suffix}.mp4") as on_frame:
        env = RobotEnv(model, motion_for_arm(motion, arm), on_frame=on_frame)
        if is_flip:
            result = build_flipped_tower(
                env,
                scene,
                triangle,
                flip_for_arm(load_flip_config(path), arm),
                GLASS_NAMES,
                FLIPPED_NAME,
                build_glass_triangle,
            )
        else:
            result = build_glass_triangle(env, scene, triangle, GLASS_NAMES, arms, carries)
            if carries:
                report["hand_height_limited"] = [round(v, 2) for v in result.carry_limited]
    lower = [env.object_position(name)[:2] for name in GLASS_NAMES[:2]]
    built = (lower[0] + lower[1]) / 2
    report.update(
        success=result.success,
        failures=list(result.failures),
        failed_at=first_failure(result.steps),
        steps=step_summary(result.steps),
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
    parser.add_argument(
        "--stay-in-middle",
        action="store_true",
        help="for a flip, keep the robot mid-table with its right arm and move nothing",
    )
    parser.add_argument(
        "--two-arms",
        action="store_true",
        help="for a triangle, use each arm for the glasses on its own side",
    )
    parser.add_argument(
        "--hand-path",
        action="store_true",
        help="for a triangle, carry each glass along the path of the hand in the video",
    )
    parser.add_argument(
        "--instruction",
        help='a typed instruction, read by a vision language model, e.g. "nearest at the bottom"',
    )
    parser.add_argument(
        "--name", default="", help="added to output file names, to keep runs apart"
    )
    parser.add_argument(
        "--find-glasses",
        action="store_true",
        help="ask a vision language model for glasses the reader may have missed",
    )
    parser.add_argument(
        "--sam-tracks",
        action="store_true",
        help="read the glasses from SAM 2's saved tracks (run scripts/sam_track_glasses.py first)",
    )
    parser.add_argument(
        "--copy-side-order",
        action="store_true",
        help="for a triangle, put the first lower glass on the side the person did",
    )
    parser.add_argument(
        "--choose-arms",
        action="store_true",
        help="for a triangle, let the outcome model choose one arm or two",
    )
    args = parser.parse_args()
    raw = yaml.safe_load(CONFIG_PATH.read_text())["real_video"]
    config = GlassVideoConfig(**raw)
    try:
        points = glasses_from_model(args.video) if args.find_glasses else None
        finals = glasses_near_end(args.video) if args.find_glasses else None
    except InstructionError as error:
        logger.error("Couldn't ask the vision model for glasses: %s", error)
        raise SystemExit(1) from error
    try:
        tracks_path = ROOT / "data/processed" / args.video.stem / "sam_tracks.json"
        sam = json.loads(tracks_path.read_text()) if args.sam_tracks else None
        demo, is_flip = read_video(args.video, config, points, finals, sam)
    except VideoReadError as error:
        logger.error("Couldn't read %s: %s", args.video.name, error)
        raise SystemExit(1) from error
    left_out = None
    instruction_report = None
    if args.instruction:
        if args.hand_path:
            raise SystemExit("--instruction can't be combined with --hand-path yet")
        try:
            demo, is_flip, left_out, instruction_report = follow_instruction(
                args.video, demo, is_flip, args.instruction
            )
        except InstructionError as error:
            logger.error("Couldn't follow the instruction: %s", error)
            raise SystemExit(1) from error
    suffix_extra = f"_{args.name}" if args.name else ("_instruction" if args.instruction else "")
    report = run_robot(
        args.video,
        demo,
        is_flip,
        args.fit_to_robot,
        args.stay_in_middle,
        config,
        args.two_arms,
        args.hand_path,
        left_out,
        suffix_extra,
        {"instruction": instruction_report} if instruction_report else None,
        args.choose_arms,
        args.copy_side_order,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
