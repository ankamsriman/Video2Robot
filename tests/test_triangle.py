"""What counts as a real triangle of cups, measured from where the cups end up."""

import numpy as np
import pytest

from src.verification.triangle import (
    CupPose,
    TriangleConfig,
    triangle_failures,
    triangle_targets,
)

CONFIG = TriangleConfig(
    center=(0.47, -0.13),
    cup_spacing=0.085,
    cup_height=0.09,
    rim_radius=0.04,
    top_radius=0.028,
    position_tolerance=0.02,
    max_tilt_degrees=10.0,
)
TABLE = 0.72


def _upright(x: float, y: float, level: int) -> CupPose:
    z = TABLE + CONFIG.cup_height * (level + 0.5)
    return CupPose(position=(x, y, z), up=(0.0, 0.0, 1.0))


def _good_triangle() -> tuple[CupPose, CupPose, CupPose]:
    return (
        _upright(0.47, -0.0875, 0),
        _upright(0.47, -0.1725, 0),
        _upright(0.47, -0.13, 1),
    )


def test_targets_put_two_cups_side_by_side_and_one_above_the_middle() -> None:
    left, right, top = triangle_targets(CONFIG, TABLE)

    assert left == pytest.approx((0.47, -0.0875, TABLE + 0.045))
    assert right == pytest.approx((0.47, -0.1725, TABLE + 0.045))
    assert top == pytest.approx((0.47, -0.13, TABLE + 0.135))


def test_top_cup_rim_reaches_over_both_lower_cups() -> None:
    left, right, top = triangle_targets(CONFIG, TABLE)
    inner_edge = abs(left[1] - top[1]) - CONFIG.top_radius

    assert inner_edge < CONFIG.rim_radius
    assert abs(right[1] - top[1]) - CONFIG.top_radius < CONFIG.rim_radius


def test_a_good_triangle_has_no_failures() -> None:
    assert triangle_failures(*_good_triangle(), CONFIG, TABLE) == ()


def test_top_cup_on_the_table_is_not_a_triangle() -> None:
    left, right, _ = _good_triangle()
    fallen = _upright(0.47, -0.13, 0)

    assert "top cup is not resting on the lower cups" in triangle_failures(
        left, right, fallen, CONFIG, TABLE
    )


def test_a_tipped_cup_is_not_a_triangle() -> None:
    left, right, top = _good_triangle()
    tilt = np.radians(30)
    tipped = CupPose(top.position, (float(np.sin(tilt)), 0.0, float(np.cos(tilt))))

    assert "top cup is tilted" in triangle_failures(left, right, tipped, CONFIG, TABLE)


def test_lower_cups_too_far_apart_are_not_a_triangle() -> None:
    left, _, top = _good_triangle()
    far_right = _upright(0.47, -0.30, 0)

    assert "lower cups are not side by side" in triangle_failures(
        left, far_right, top, CONFIG, TABLE
    )


def test_top_cup_off_to_one_side_is_not_a_triangle() -> None:
    left, right, _ = _good_triangle()
    offset = _upright(0.47, -0.08, 1)

    assert "top cup is not centred over both" in triangle_failures(
        left, right, offset, CONFIG, TABLE
    )


@pytest.mark.integration
def test_g1_builds_the_triangle_in_physics() -> None:
    from scripts.cup_trial import run_trial

    result = run_trial((1, False, "configs/cups.yaml"))

    assert result["success"], result
