"""The complete chain, run on a rendered demo video with known positions."""

from pathlib import Path

import numpy as np
import pytest

from src.perception.observe_video import load_marker_config
from src.pipeline.run_video import (
    paths_for,
    run_perception,
    run_planning,
    run_simulation,
)
from src.pipeline.synthetic_demo import render_synthetic_demo
from src.schemas.plan import Plan, Relation, Step
from src.simulation.scene import load_scene_config

TRUTH = Plan(
    steps=(
        Step(object="lemon", reference="box", relation=Relation.IN),
        Step(object="lime", reference="bowl", relation=Relation.IN),
        Step(object="apple", reference="box", relation=Relation.IN),
    )
)
STARTS = {
    "apple": (0.36, -0.27),
    "lemon": (0.34, -0.15),
    "lime": (0.35, -0.03),
    "box": (0.50, -0.21),
    "bowl": (0.50, 0.03),
}
TARGETS = {"apple": (0.50, -0.26), "lemon": (0.50, -0.16), "lime": (0.50, 0.03)}


@pytest.mark.integration
def test_rendered_video_drives_the_robot_to_the_correct_result(tmp_path: Path) -> None:
    video = tmp_path / "demo.mp4"
    render_synthetic_demo(
        load_scene_config(), load_marker_config(), TRUTH, STARTS, TARGETS, video
    )
    paths = paths_for(video, tmp_path)

    run_perception(video, paths)
    demo = run_planning(paths)
    result = run_simulation(paths)

    assert demo.plan == TRUTH
    for name in ("apple", "lemon", "lime", "box"):
        error = np.linalg.norm(
            np.array(demo.start_positions[name]) - np.array(STARTS[name])
        )
        assert error < 0.03, name
    assert result["success"], result
    assert paths.rollout.exists()
