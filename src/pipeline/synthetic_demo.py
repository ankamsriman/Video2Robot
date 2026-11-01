"""Render a demo video where every true position is known, to test the vision chain.

There's no hand and no robot in it: objects glide through the air to their targets. It
checks tracking, the marker mapping and plan-making, not hand detection.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np

from src.perception.homography import MarkerConfig
from src.schemas.plan import Plan
from src.simulation.scene import SceneConfig

WIDTH, HEIGHT = 1280, 720
FPS = 30
MARKER_CELLS = 6
PAGE_MARGIN = 0.03
SURFACE_THICKNESS = 0.0005
CARRY_HEIGHT = 0.12
MOVE_SECONDS = 1.5
PAUSE_SECONDS = 1.5
Position = tuple[float, float]


@dataclass(frozen=True)
class CameraView:
    """Where the phone would be: on the far side of the table, looking down."""

    lookat: tuple[float, float, float] = (0.42, -0.10, 0.72)
    # About 75 cm above the tabletop, which a short tripod on a second table can reach.
    distance: float = 0.98
    azimuth: float = 180.0
    elevation: float = -50.0


def _add_marker(body: mujoco.MjsBody, marker: MarkerConfig, height: float) -> None:
    """Build the printed page and its black and white squares from flat geoms."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    bits = cv2.aruco.generateImageMarker(dictionary, marker.marker_id, MARKER_CELLS)
    cell = marker.side / MARKER_CELLS
    x_center, y_center = marker.center
    page = marker.side / 2 + PAGE_MARGIN
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[x_center, y_center, height],
        size=[page, page, SURFACE_THICKNESS],
        rgba=[1, 1, 1, 1],
        contype=0,
        conaffinity=0,
    )
    for row in range(MARKER_CELLS):
        for column in range(MARKER_CELLS):
            if bits[row, column] > 127:
                continue
            # The image's top row is furthest from the robot; its left is the robot's left.
            x = x_center + marker.side / 2 - (row + 0.5) * cell
            y = y_center + marker.side / 2 - (column + 0.5) * cell
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, y, height + 2 * SURFACE_THICKNESS],
                size=[cell / 2, cell / 2, SURFACE_THICKNESS],
                rgba=[0, 0, 0, 1],
                contype=0,
                conaffinity=0,
            )


def _build_scene(
    config: SceneConfig,
    marker: MarkerConfig,
    positions: Mapping[str, Position],
) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    world = spec.worldbody
    world.add_light(pos=[0.4, -0.1, 2.0], dir=[0, 0, -1], castshadow=False)
    top = config.table.top_height
    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.45, -0.1, top - 0.01],
        size=[0.6, 0.7, 0.01],
        rgba=[0.55, 0.55, 0.55, 1],
    )
    _add_marker(world, marker, top + SURFACE_THICKNESS)
    for item in config.containers:
        half_x, half_y, height = item.size[0] / 2, item.size[1] / 2, item.size[2]
        body = world.add_body(name=item.name, pos=[*positions[item.name], top])
        wall = 0.004
        for size, offset in (
            ([half_x, half_y, wall], [0, 0, wall]),
            ([wall, half_y, height / 2], [half_x - wall, 0, height / 2]),
            ([wall, half_y, height / 2], [-half_x + wall, 0, height / 2]),
            ([half_x, wall, height / 2], [0, half_y - wall, height / 2]),
            ([half_x, wall, height / 2], [0, -half_y + wall, height / 2]),
        ):
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=offset, rgba=item.rgba
            )
    for item in config.objects:
        half = [length / 2 for length in item.size]
        body = world.add_body(name=item.name, mocap=True, pos=[0, 0, 0])
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half,
            rgba=item.rgba,
            contype=0,
            conaffinity=0,
        )
    model = spec.compile()
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    return model


def _carry_path(
    start: np.ndarray, end: np.ndarray, frames: int, rest_height: float
) -> np.ndarray:
    progress = np.linspace(0, 1, frames)[:, None]
    path = start + progress * (end - start)
    lift = CARRY_HEIGHT * np.sin(np.pi * progress)
    return np.column_stack((path[:, 0], path[:, 1], rest_height + lift[:, 0]))


def render_synthetic_demo(
    config: SceneConfig,
    marker: MarkerConfig,
    plan: Plan,
    starts: Mapping[str, Position],
    targets: Mapping[str, Position],
    output: Path,
    view: CameraView | None = None,
) -> None:
    """Write a video in which each planned object travels from its start to its target."""
    view = view or CameraView()
    model = _build_scene(config, marker, starts)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = view.lookat
    camera.distance = view.distance
    camera.azimuth = view.azimuth
    camera.elevation = view.elevation
    top = config.table.top_height
    rest = {item.name: top + item.size[2] / 2 for item in config.objects}
    positions = {
        item.name: np.array([*starts[item.name], rest[item.name]])
        for item in config.objects
    }
    mocap_ids = {
        item.name: model.body_mocapid[model.body(item.name).id]
        for item in config.objects
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    with (
        mujoco.Renderer(model, HEIGHT, WIDTH) as renderer,
        imageio.get_writer(output, fps=FPS, macro_block_size=1) as writer,
    ):

        def write_frame() -> None:
            for name, mocap in mocap_ids.items():
                data.mocap_pos[mocap] = positions[name]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            writer.append_data(renderer.render())

        for _ in range(int(PAUSE_SECONDS * FPS)):
            write_frame()
        for step in plan.steps:
            start = positions[step.object][:2]
            end = np.array(targets[step.object])
            frames = int(MOVE_SECONDS * FPS)
            for point in _carry_path(start, end, frames, rest[step.object]):
                positions[step.object] = point
                write_frame()
            for _ in range(int(PAUSE_SECONDS * FPS)):
                write_frame()
