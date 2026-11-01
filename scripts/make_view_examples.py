"""Draw what the phone should see for each task, from the real measurements.

These are simulator pictures, not photos. Each one is checked by running the same marker
detector that real videos go through.

PYTHONPATH=. .venv/bin/python scripts/make_view_examples.py
"""

import json
import logging
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from src.perception.homography import MarkerNotFoundError, estimate_mapping
from src.perception.observe_video import load_marker_config
from src.pipeline.synthetic_demo import _add_marker
from src.simulation.scene import ROOT

logger = logging.getLogger("make_view_examples")
OUTPUT = ROOT / "docs/recording_views"
WIDTH, HEIGHT = 1280, 720
PHONE_FIELD_OF_VIEW = 42.0
TABLE_TOP = 0.72
TABLE_CENTER = (0.46, -0.10)
TABLE_HALF = (0.30, 0.45)
MARKER = load_marker_config()
PHONE_ABOVE_TABLE = 0.75
PHONE_ELEVATION = 50.0
LOOK_AT = (0.42, -0.10, TABLE_TOP)
RED, YELLOW, GREEN, BLUE = (
    (0.80, 0.15, 0.15, 1),
    (0.92, 0.78, 0.15, 1),
    (0.20, 0.65, 0.25, 1),
    (0.20, 0.35, 0.80, 1),
)
TRAY_LIGHT, TRAY_DARK = (0.85, 0.85, 0.90, 1), (0.45, 0.30, 0.18, 1)
BLOCK = (0.06, 0.07, 0.07)
TRAY = (0.15, 0.18, 0.04)
CUP_HEIGHT, CUP_WIDE, CUP_NARROW = 0.09, 0.04, 0.028
CUP_SIDES = 48
LOWER_GAP = 0.0425


@dataclass(frozen=True)
class Block:
    label: str
    position: tuple[float, float]
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class Tray:
    label: str
    position: tuple[float, float]
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class Cup:
    label: str
    position: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    flipped: bool = False


@dataclass(frozen=True)
class View:
    name: str
    title: str
    items: tuple[Block | Tray | Cup, ...]


def _table(world: mujoco.MjsBody) -> None:
    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[*TABLE_CENTER, TABLE_TOP - 0.01],
        size=[*TABLE_HALF, 0.01],
        rgba=[0.62, 0.48, 0.34, 1],
    )


def _add_item(world: mujoco.MjsBody, item: Block | Tray | Cup) -> None:
    if isinstance(item, Block):
        half = [length / 2 for length in BLOCK]
        world.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[*item.position, TABLE_TOP + half[2]],
            size=half,
            rgba=item.rgba,
        )
    elif isinstance(item, Tray):
        half_x, half_y, height = TRAY[0] / 2, TRAY[1] / 2, TRAY[2]
        wall = 0.004
        for size, offset in (
            ([half_x, half_y, wall], [0, 0, wall]),
            ([wall, half_y, height / 2], [half_x - wall, 0, height / 2]),
            ([wall, half_y, height / 2], [-half_x + wall, 0, height / 2]),
            ([half_x, wall, height / 2], [0, half_y - wall, height / 2]),
            ([half_x, wall, height / 2], [0, -half_y + wall, height / 2]),
        ):
            world.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[
                    item.position[0] + offset[0],
                    item.position[1] + offset[1],
                    TABLE_TOP + offset[2],
                ],
                size=size,
                rgba=item.rgba,
            )
    else:
        _add_cup(world, item)


def _cup_mesh(spec: mujoco.MjSpec, name: str, wide_at_top: bool) -> None:
    """A tapered cup shape, centred on its middle."""
    bottom, top = (CUP_NARROW, CUP_WIDE) if wide_at_top else (CUP_WIDE, CUP_NARROW)
    angles = np.linspace(0, 2 * np.pi, CUP_SIDES, endpoint=False)
    vertices = [
        (radius * np.cos(angle), radius * np.sin(angle), z)
        for radius, z in ((bottom, -CUP_HEIGHT / 2), (top, CUP_HEIGHT / 2))
        for angle in angles
    ]
    mesh = spec.add_mesh()
    mesh.name = name
    mesh.uservert = np.array(vertices).ravel().tolist()


def _add_cup(world: mujoco.MjsBody, cup: Cup) -> None:
    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="cup_up" if cup.flipped else "cup_down",
        pos=list(cup.position),
        rgba=cup.rgba,
    )
    if cup.flipped:
        x, y, z = cup.position
        world.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[x, y, z + CUP_HEIGHT / 2 + 0.0005],
            size=[CUP_WIDE * 0.85, 0.0005, 0],
            rgba=[0.05, 0.05, 0.08, 1],
        )


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = LOOK_AT
    camera.distance = PHONE_ABOVE_TABLE / np.sin(np.radians(PHONE_ELEVATION))
    camera.azimuth = 180.0
    camera.elevation = -PHONE_ELEVATION
    return camera


def _phone_position() -> np.ndarray:
    distance = PHONE_ABOVE_TABLE / np.tan(np.radians(PHONE_ELEVATION))
    return np.array([LOOK_AT[0] + distance, LOOK_AT[1], TABLE_TOP + PHONE_ABOVE_TABLE])


def _project(point: tuple[float, float, float]) -> tuple[int, int]:
    """Where a 3D point appears in the picture."""
    phone = _phone_position()
    forward = np.array(LOOK_AT) - phone
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    ray = np.array(point) - phone
    focal = HEIGHT / 2 / np.tan(np.radians(PHONE_FIELD_OF_VIEW) / 2)
    depth = ray @ forward
    return (
        int(WIDTH / 2 + focal * (ray @ right) / depth),
        int(HEIGHT / 2 - focal * (ray @ up) / depth),
    )


def _label_at(item: Block | Tray | Cup) -> tuple[float, float, float]:
    if isinstance(item, Block):
        return (*item.position, TABLE_TOP + BLOCK[2])
    if isinstance(item, Tray):
        return (item.position[0], item.position[1], TABLE_TOP + TRAY[2])
    return (item.position[0], item.position[1], item.position[2] + CUP_HEIGHT / 2)


def _text(
    image: np.ndarray, text: str, at: tuple[int, int], scale: float = 0.5
) -> None:
    """Write a label on a dark plate, centred on `at`."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), _ = cv2.getTextSize(text, font, scale, 1)
    x, y = max(at[0] - width // 2, 4), at[1]
    cv2.rectangle(
        image, (x - 4, y - height - 4), (x + width + 4, y + 5), (25, 25, 25), -1
    )
    cv2.putText(image, text, (x, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _render(view: View) -> np.ndarray:
    spec = mujoco.MjSpec()
    world = spec.worldbody
    world.add_light(pos=[0.5, -0.1, 2.0], dir=[0, 0, -1], castshadow=False)
    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.32, 0.34, 0.38, 1]
    )
    _cup_mesh(spec, "cup_down", wide_at_top=False)
    _cup_mesh(spec, "cup_up", wide_at_top=True)
    _table(world)
    _add_marker(world, MARKER, TABLE_TOP + 0.0005)
    for item in view.items:
        _add_item(world, item)
    model = spec.compile()
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    model.vis.global_.fovy = PHONE_FIELD_OF_VIEW
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        renderer.update_scene(data, _camera())
        return renderer.render().copy()


def _marker_seen(picture: np.ndarray) -> bool:
    try:
        estimate_mapping([cv2.cvtColor(picture, cv2.COLOR_RGB2BGR)], MARKER)
    except MarkerNotFoundError:
        return False
    return True


def _in_frame(view: View) -> bool:
    points = [_label_at(item) for item in view.items]
    points += [(MARKER.center[0], MARKER.center[1], TABLE_TOP)]
    return all(
        0 <= x < WIDTH and 0 <= y < HEIGHT
        for x, y in (_project(point) for point in points)
    )


def _annotate(picture: np.ndarray, view: View) -> np.ndarray:
    image = cv2.cvtColor(picture, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (0, 0), (WIDTH, 36), (30, 30, 30), -1)
    cv2.putText(
        image,
        view.title,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    x, y = _project((*MARKER.center, TABLE_TOP))
    _text(image, "marker: black square 10 cm", (x + 210, y + 5))
    for item in view.items:
        if item.label:
            x, y = _project(_label_at(item))
            _text(image, item.label, (x, y - 34 if isinstance(item, Block) else y - 14))
    _text(image, "YOU stand at this end", (WIDTH // 2, 62))
    _text(
        image,
        "PHONE is at this end, about 75 cm above the tabletop",
        (WIDTH // 2, HEIGHT - 12),
    )
    return image


def views() -> tuple[View, ...]:
    tower_x, tower_y = 0.47, -0.13
    lower = TABLE_TOP + CUP_HEIGHT / 2
    flip_x, flip_y = 0.365, -0.30
    trays = (
        Tray("tray 15 x 18 cm", (0.50, 0.03), TRAY_LIGHT),
        Tray("tray 15 x 18 cm", (0.50, -0.21), TRAY_DARK),
    )
    return (
        View(
            "task1_start",
            "Task 1 sorting: first frame",
            (
                *trays,
                Block("7 cm block", (0.35, -0.28), RED),
                Block("7 cm block", (0.35, -0.16), YELLOW),
                Block("7 cm block", (0.35, -0.04), GREEN),
            ),
        ),
        View(
            "task1_end",
            "Task 1 sorting: last frame",
            (
                *trays,
                Block("", (0.50, -0.26), RED),
                Block("", (0.50, -0.16), GREEN),
                Block("", (0.50, 0.03), YELLOW),
            ),
        ),
        View(
            "task2_start",
            "Task 2 triangle: first frame",
            (
                Cup("7-8 cm cup", (0.33, -0.05, lower), YELLOW),
                Cup("7-8 cm cup", (0.33, -0.17, lower), GREEN),
                Cup("7-8 cm cup", (0.33, -0.29, lower), RED),
            ),
        ),
        View(
            "task2_end",
            "Task 2 triangle: last frame",
            (
                Cup("", (tower_x, tower_y + LOWER_GAP, lower), YELLOW),
                Cup("", (tower_x, tower_y - LOWER_GAP, lower), GREEN),
                Cup("", (tower_x, tower_y, lower + CUP_HEIGHT), RED),
            ),
        ),
        View(
            "task3_start",
            "Task 3 flip: first frame",
            (
                Cup("7-8 cm cup", (0.33, 0.14, lower), YELLOW),
                Cup("7-8 cm cup", (0.33, 0.02, lower), GREEN),
                Cup("7-8 cm cup", (0.34, -0.10, lower), RED),
                Cup("fourth cup, opening down", (0.44, 0.07, lower), BLUE),
            ),
        ),
        View(
            "task3_end",
            "Task 3 flip: last frame",
            (
                Cup("", (flip_x, flip_y + LOWER_GAP, lower), YELLOW),
                Cup("", (flip_x, flip_y - LOWER_GAP, lower), GREEN),
                Cup("", (flip_x, flip_y, lower + CUP_HEIGHT), RED),
                Cup("opening up", (flip_x, flip_y, lower + 2 * CUP_HEIGHT), BLUE, True),
            ),
        ),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for view in views():
        picture = _render(view)
        results[view.name] = {
            "marker_detected": _marker_seen(picture),
            "everything_in_frame": _in_frame(view),
        }
        cv2.imwrite(str(OUTPUT / f"{view.name}.png"), _annotate(picture, view))
    logger.info(json.dumps(results, indent=1))
    logger.info("Phone position: %s", np.round(_phone_position(), 2).tolist())


if __name__ == "__main__":
    main()
