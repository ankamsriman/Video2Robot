"""Save simulation runs to video, writing frames straight to disk to save memory."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import imageio
import mujoco

from src.simulation.robot_env import FrameCallback

WIDTH = 960
HEIGHT = 720
FPS = 30
CAMERA = {
    "lookat": (0.30, -0.02, 0.82),
    "distance": 1.45,
    "azimuth": -130.0,
    "elevation": -30.0,
}


@contextmanager
def video_recorder(model: mujoco.MjModel, path: Path) -> Iterator[FrameCallback]:
    """Provide a frame callback for RobotEnv that appends rendered frames to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = CAMERA["lookat"]
    camera.distance = CAMERA["distance"]
    camera.azimuth = CAMERA["azimuth"]
    camera.elevation = CAMERA["elevation"]

    with (
        mujoco.Renderer(model, HEIGHT, WIDTH) as renderer,
        imageio.get_writer(path, fps=FPS, macro_block_size=1) as writer,
    ):

        def on_frame(data: mujoco.MjData) -> None:
            renderer.update_scene(data, camera)
            writer.append_data(renderer.render())

        yield on_frame
