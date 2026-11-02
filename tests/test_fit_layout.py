"""Fitting a recorded layout into the area where the robot's flip works."""

import numpy as np
import pytest

from src.reasoning.fit_layout import Area, fit_flip_layout, plan_stand

STARTS = Area((0.30, 0.40), (-0.14, 0.14))
FLIP = Area((0.40, 0.46), (-0.10, 0.05))
TOWER = Area((0.36, 0.38), (-0.32, -0.28))
GAP = 0.12


def _fit(starts: list[tuple[float, float]], tower: tuple[float, float]):
    return fit_flip_layout(
        [np.array(s) for s in starts], np.array(tower), STARTS, FLIP, TOWER, GAP, 0.15
    )


def test_a_layout_already_inside_stays_exactly_where_it_is() -> None:
    starts = [(0.30, 0.12), (0.30, 0.0), (0.30, -0.12), (0.44, 0.0)]

    fitted = _fit(starts, (0.37, -0.30))

    assert fitted.tower_shift == pytest.approx(0.0)
    assert np.allclose(fitted.start_shifts, 0.0, atol=0.011)


def test_a_far_tower_moves_to_the_nearest_allowed_spot_and_reports_it() -> None:
    fitted = _fit([(0.30, 0.12), (0.30, 0.0), (0.30, -0.12), (0.44, 0.0)], (0.41, 0.05))

    assert np.allclose(fitted.tower, (0.38, -0.28))
    assert fitted.tower_shift == pytest.approx(np.hypot(0.03, 0.33))


def test_fitted_glasses_keep_their_distance_from_each_other() -> None:
    crowded = [(0.35, 0.0), (0.351, 0.0), (0.352, 0.0), (0.45, 0.0)]

    fitted = _fit(crowded, (0.37, -0.30))

    points = fitted.starts
    for i in range(4):
        for j in range(i):
            assert np.linalg.norm(points[i] - points[j]) >= GAP - 1e-9


def test_a_layout_with_no_room_is_refused() -> None:
    tiny = Area((0.30, 0.31), (0.0, 0.01))
    with pytest.raises(ValueError, match="No free spot"):
        fit_flip_layout(
            [np.array((0.3, 0.0))] * 4,
            np.array((0.37, -0.30)),
            tiny,
            FLIP,
            TOWER,
            GAP,
            0.15,
        )


def _stand(starts: list[tuple[float, float]], tower: tuple[float, float]):
    return plan_stand(
        [np.array(s) for s in starts], np.array(tower), 0.285, table_half_width=0.40
    )


def test_a_tower_left_of_the_glasses_calls_for_the_left_arm() -> None:
    plan = _stand([(0.32, -0.19), (0.39, -0.25), (0.46, -0.16), (0.49, -0.26)], (0.41, 0.05))

    assert plan.arm == "left"
    assert plan.robot_y == pytest.approx(0.05 - 0.285)


def test_a_tower_right_of_the_glasses_calls_for_the_right_arm() -> None:
    plan = _stand([(0.32, 0.1), (0.35, 0.0), (0.34, -0.1), (0.44, 0.0)], (0.40, -0.29))

    assert plan.arm == "right"
    assert plan.robot_y == pytest.approx(-0.29 + 0.285)


def test_the_robot_never_stands_past_the_end_of_the_table() -> None:
    plan = _stand([(0.32, 0.3), (0.35, 0.3), (0.34, 0.3), (0.44, 0.3)], (0.40, -0.35))

    assert plan.arm == "right"
    assert plan.robot_y == pytest.approx(-0.35 + 0.285)
    far = _stand([(0.32, -0.3), (0.35, -0.3), (0.34, -0.3), (0.44, -0.3)], (0.40, -0.15))
    assert far.arm == "left"
    assert far.robot_y == pytest.approx(-0.40)
