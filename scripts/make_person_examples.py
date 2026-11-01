"""Draw where the person's body and hands should appear in the phone's view.

These are simulator pictures with a simple model of a person, not photos. Each one is
checked by running the same marker detector that real videos go through.

PYTHONPATH=. .venv/bin/python scripts/make_person_examples.py
"""

import json
import logging
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from scripts.make_view_examples import (
    BLOCK,
    GREEN,
    OUTPUT,
    RED,
    TABLE_TOP,
    TRAY_DARK,
    TRAY_LIGHT,
    WIDTH,
    YELLOW,
    Block,
    Tray,
    View,
    _marker_seen,
    _text,
    render_view,
)

logger = logging.getLogger("make_person_examples")
SKIN = (0.78, 0.58, 0.45, 1)
SHIRT = (0.16, 0.17, 0.20, 1)
TRAYS = (Tray("", (0.50, 0.03), TRAY_LIGHT), Tray("", (0.50, -0.21), TRAY_DARK))
SHOULDER = np.array([0.03, -0.30, TABLE_TOP + 0.42])
ELBOW = np.array([0.10, -0.34, TABLE_TOP + 0.20])
KNUCKLE_SPACING = 0.024
PALM_LENGTH = 0.08
GOOD_BAR, BAD_BAR = (40, 120, 40), (40, 40, 170)


@dataclass(frozen=True)
class PersonView:
    name: str
    title: str
    is_good: bool
    items: tuple[Block | Tray, ...]
    wrist: tuple[float, float, float] | None
    held_block: tuple[float, float, float] | None
    labels: tuple[tuple[str, tuple[int, int]], ...]


def _capsule(
    world: mujoco.MjsBody,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[*start, *end],
        size=[radius, 0, 0],
        rgba=rgba,
    )


def _add_body(world: mujoco.MjsBody) -> None:
    """The torso, with the head well above the top of the phone's view."""
    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.02, -0.10, TABLE_TOP + 0.05],
        size=[0.09, 0.20, 0.55],
        rgba=SHIRT,
    )
    _capsule(
        world,
        np.array([0.02, -0.10, TABLE_TOP + 0.62]),
        np.array([0.02, -0.10, TABLE_TOP + 0.72]),
        0.09,
        SKIN,
    )


def _add_arm_and_hand(world: mujoco.MjsBody, wrist: np.ndarray) -> None:
    """A right arm reaching forward, fingers pointing towards the phone and down."""
    _capsule(world, SHOULDER, ELBOW, 0.04, SHIRT)
    _capsule(world, ELBOW, wrist, 0.03, SKIN)
    forward = np.array([0.97, 0.0, -0.24])
    side = np.array([0.0, 1.0, 0.0])
    for index, length in zip((-1.5, -0.5, 0.5, 1.5), (0.055, 0.07, 0.075, 0.07)):
        offset = side * index * KNUCKLE_SPACING
        knuckle = wrist + forward * PALM_LENGTH + offset
        _capsule(world, wrist + offset * 0.6, knuckle, 0.015, SKIN)
        _capsule(world, knuckle, knuckle + forward * length, 0.009, SKIN)
    thumb_base = wrist + forward * 0.03 + side * 0.04
    thumb_tip = thumb_base + forward * 0.05 + side * 0.03 + np.array([0, 0, -0.02])
    _capsule(world, thumb_base, thumb_tip, 0.011, SKIN)


def _scene_extras(view: PersonView):
    def add(world: mujoco.MjsBody) -> None:
        _add_body(world)
        if view.wrist is not None:
            _add_arm_and_hand(world, np.array(view.wrist))
        if view.held_block is not None:
            world.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=list(view.held_block),
                size=[length / 2 for length in BLOCK],
                rgba=RED,
            )

    return add


def _annotate(picture: np.ndarray, view: PersonView) -> np.ndarray:
    image = cv2.cvtColor(picture, cv2.COLOR_RGB2BGR)
    bar = GOOD_BAR if view.is_good else BAD_BAR
    cv2.rectangle(image, (0, 0), (WIDTH, 36), bar, -1)
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
    for text, at in view.labels:
        _text(image, text, at)
    return image


def views() -> tuple[PersonView, ...]:
    table = TABLE_TOP
    block_top = table + BLOCK[2]
    red = Block("", (0.35, -0.28), RED)
    yellow = Block("", (0.35, -0.16), YELLOW)
    green = Block("", (0.35, -0.04), GREEN)
    start_items = (*TRAYS, red, yellow, green)
    picked_items = (*TRAYS, yellow, green)
    marker_clear = ("Marker fully clear", (850, 560))
    return (
        PersonView(
            "person_1_start",
            "GOOD - first and last 2 seconds: hands off the table",
            True,
            start_items,
            None,
            None,
            (
                ("Only your waist shows. Your head stays out of the shot.", (640, 90)),
                ("Hands down at your sides", (640, 165)),
            ),
        ),
        PersonView(
            "person_2_pick",
            "GOOD - picking: pause here for a moment",
            True,
            start_items,
            (0.24, -0.28, block_top + 0.045),
            None,
            (
                ("Whole hand and all fingers in view", (260, 270)),
                ("Grip from above or the sides, so the front stays visible", (900, 150)),
                marker_clear,
            ),
        ),
        PersonView(
            "person_3_place",
            "GOOD - placing: lower it in, pause, then lift your hand away",
            True,
            picked_items,
            (0.37, -0.21, table + 0.165),
            (0.50, -0.21, table + 0.08),
            (
                ("Arm comes from YOUR side, over the table", (240, 150)),
                marker_clear,
            ),
        ),
        PersonView(
            "person_4_mistake",
            "MISTAKE - hand over the marker: the system loses the table",
            False,
            start_items,
            (0.55, -0.10, table + 0.09),
            None,
            (("Never reach past the trays", (240, 345)),),
        ),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = {}
    for view in views():
        picture = render_view(
            View(view.name, view.title, view.items), _scene_extras(view)
        )
        results[view.name] = {"marker_detected": _marker_seen(picture)}
        cv2.imwrite(str(OUTPUT / f"{view.name}.png"), _annotate(picture, view))
    logger.info(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
