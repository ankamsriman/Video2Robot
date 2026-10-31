"""Build the digital twin: the G1 standing at a table with objects and containers."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/default.yaml"

# Midpoint between the thumb and fingertips, in the wrist frame (calibrated in Phase 1).
GRASP_POINT_IN_WRIST = (0.125, 0.08, 0.0)
WALL_THICKNESS = 0.006


@dataclass(frozen=True)
class ItemConfig:
    """One movable object or container. `size` is full width, depth and height."""

    name: str
    size: tuple[float, float, float]
    position: tuple[float, float]
    rgba: tuple[float, float, float, float]
    mass: float = 0.0


@dataclass(frozen=True)
class TableConfig:
    near_edge_x: float
    depth: float
    width: float
    top_height: float


@dataclass(frozen=True)
class SceneConfig:
    g1_scene: Path
    table: TableConfig
    objects: tuple[ItemConfig, ...]
    containers: tuple[ItemConfig, ...]


def _read_items(section: Mapping[str, Mapping]) -> tuple[ItemConfig, ...]:
    return tuple(
        ItemConfig(
            name=name,
            size=tuple(item["size"]),
            position=tuple(item["position"]),
            rgba=tuple(item["rgba"]),
            mass=item.get("mass", 0.0),
        )
        for name, item in section.items()
    )


def load_scene_config(path: Path = CONFIG_PATH) -> SceneConfig:
    raw = yaml.safe_load(path.read_text())
    scene = raw["scene"]
    return SceneConfig(
        g1_scene=ROOT / raw["paths"]["g1_scene"],
        table=TableConfig(**scene["table"]),
        objects=_read_items(scene["objects"]),
        containers=_read_items(scene["containers"]),
    )


def _add_table(spec: mujoco.MjSpec, table: TableConfig) -> None:
    thickness = 0.02
    center_x = table.near_edge_x + table.depth / 2
    body = spec.worldbody.add_body(
        name="table", pos=[center_x, 0, table.top_height - thickness]
    )
    body.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[table.depth / 2, table.width / 2, thickness],
        rgba=[0.55, 0.42, 0.30, 1],
        friction=[1.0, 0.01, 0.001],
    )


def _add_object(spec: mujoco.MjSpec, item: ItemConfig, table_top: float) -> None:
    half = [length / 2 for length in item.size]
    body = spec.worldbody.add_body(
        name=item.name, pos=[*item.position, table_top + half[2] + 0.001]
    )
    body.add_freejoint(name=f"{item.name}_free")
    body.add_geom(
        name=f"{item.name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half,
        mass=item.mass,
        rgba=item.rgba,
        friction=[1.5, 0.02, 0.002],
        condim=4,
    )


def _add_container(spec: mujoco.MjSpec, item: ItemConfig, table_top: float) -> None:
    """A static open-top tray made of a floor and four walls."""
    half_x, half_y, height = item.size[0] / 2, item.size[1] / 2, item.size[2]
    wall = WALL_THICKNESS / 2
    body = spec.worldbody.add_body(name=item.name, pos=[*item.position, table_top])
    parts = {
        "floor": ([half_x, half_y, wall], [0, 0, wall]),
        "back": ([wall, half_y, height / 2], [half_x - wall, 0, height / 2]),
        "front": ([wall, half_y, height / 2], [-half_x + wall, 0, height / 2]),
        "left": ([half_x, wall, height / 2], [0, half_y - wall, height / 2]),
        "right": ([half_x, wall, height / 2], [0, -half_y + wall, height / 2]),
    }
    for part, (size, position) in parts.items():
        body.add_geom(
            name=f"{item.name}_{part}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=position,
            rgba=item.rgba,
        )


def _add_grasp_sites(spec: mujoco.MjSpec) -> None:
    for side, y_sign in (("right", 1.0), ("left", -1.0)):
        x, y, z = GRASP_POINT_IN_WRIST
        spec.body(f"{side}_wrist_yaw_link").add_site(
            name=f"{side}_grasp", pos=[x, y_sign * y, z], size=[0.006] * 3
        )


def _pin_pelvis(spec: mujoco.MjSpec) -> None:
    """Fix the pelvis in place, so the robot cannot lose balance. Labelled in results."""
    spec.delete(spec.body("pelvis").first_joint())


def build_model(
    config: SceneConfig,
    positions: Mapping[str, tuple[float, float]] | None = None,
    pinned: bool = True,
) -> mujoco.MjModel:
    """Compile the scene, optionally moving named objects to new table positions."""
    moved = positions or {}
    spec = mujoco.MjSpec.from_file(str(config.g1_scene))
    for key in list(spec.keys):
        spec.delete(key)
    if pinned:
        _pin_pelvis(spec)
    _add_grasp_sites(spec)
    _add_table(spec, config.table)
    for item in config.containers:
        _add_container(
            spec,
            ItemConfig(
                item.name,
                item.size,
                moved.get(item.name, item.position),
                item.rgba,
            ),
            config.table.top_height,
        )
    for item in config.objects:
        _add_object(
            spec,
            ItemConfig(
                item.name,
                item.size,
                moved.get(item.name, item.position),
                item.rgba,
                item.mass,
            ),
            config.table.top_height,
        )
    return spec.compile()
