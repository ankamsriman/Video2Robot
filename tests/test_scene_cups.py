"""A cup's shape follows the sizes given for it, so real glasses can be simulated."""

import numpy as np

from src.simulation.scene import ItemConfig, cup_vertices


def test_cup_vertices_follow_the_given_rim_and_closed_end_sizes():
    glass = ItemConfig(
        name="glass",
        size=(0.064, 0.064, 0.115),
        position=(0.0, 0.0),
        rgba=(1, 1, 1, 1),
        shape="cup",
        rim_radius=0.0365,
        top_radius=0.0275,
    )

    points = np.array(cup_vertices(glass)).reshape(-1, 3)
    radii = np.hypot(points[:, 0], points[:, 1])

    assert abs(radii[points[:, 2] < 0].max() - 0.0365) < 1e-6
    assert abs(radii[points[:, 2] > 0].max() - 0.0275) < 1e-6
    assert abs(points[:, 2].max() - 0.0575) < 1e-9
