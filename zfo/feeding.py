"""
Which food an agent can eat from its current pose.

Ported from the reach checks in wef's EFishAgent.check_and_consume_food, without the eating
cooldown, probabilistic eating or the per-step cap. Eating is an explicit action, as in
GridWorld's EAT: these functions only answer "if the agent tried to eat now, what could it get?".
Removing the pellet (arena.remove_food) and adding glucose happen in the environment.

Like movement.py and metabolism.py, these are pure functions: they take the pose and the
food positions as arguments and do not change anything.
"""

import numpy as np

from zfo.movement import wrap_angle


def food_in_reach(position, heading, food_positions, eat_radius, mouth_angle=None):
    """
    Indices into food_positions of the pellets the agent could eat, nearest first.
    A pellet is in reach if its center is within eat_radius of the agent's center and, when
    mouth_angle (radians, full width of the cone) is given, within mouth_angle / 2 of the heading.
    Returns an int array, empty if nothing is in reach.
    """
    food_positions = np.asarray(food_positions, dtype=float).reshape(-1, 2)
    offsets = food_positions - np.asarray(position, dtype=float) # offsets are arrows to each pellet from the agent
    distances = np.linalg.norm(offsets, axis=1) # get length of arrow to each pellet
    in_reach = distances < eat_radius # checks pellets that are within radius

    if mouth_angle is not None:
        angles = np.arctan2(offsets[:, 1], offsets[:, 0]) # computes the angle of the arrow
        angle_diffs = np.abs(wrap_angle(angles - heading)) # sees the difference of those angles to the agent's heading
        in_reach &= angle_diffs <= mouth_angle / 2 # that difference must be less than or equal to half the mouth angle to be in reach
        # (the mouth angle represents the size of the cone)

    indices = np.flatnonzero(in_reach)
    return indices[np.argsort(distances[indices], kind="stable")]


def nearest_food_in_reach(position, heading, food_positions, eat_radius, mouth_angle=None):
    """
    Index of the single pellet an EAT action would take (the nearest one in reach), or None.
    """
    indices = food_in_reach(position, heading, food_positions, eat_radius, mouth_angle)
    return int(indices[0]) if len(indices) else None
