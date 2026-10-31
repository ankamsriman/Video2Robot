"""Phase 0 gate: the G1 model loads, renders, and has the joints later phases rely on."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/default.yaml").read_text())


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(ROOT / CONFIG["paths"]["g1_scene"]))


def test_g1_has_arm_and_hand_joints(model: mujoco.MjModel) -> None:
    for name in (
        "right_shoulder_pitch_joint",
        "right_wrist_yaw_joint",
        "right_hand_thumb_0_joint",
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0, name


def test_g1_renders_a_non_blank_image(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    with mujoco.Renderer(model, 120, 160) as renderer:
        renderer.update_scene(data)
        image = renderer.render()

    assert image.shape == (120, 160, 3)
    assert np.std(image) > 1.0
