"""
What the agent senses about food: the continuous counterpart of GridWorld's state features.

QLearningAgent.discretize_state gave GridWorld's agent the Manhattan distance and cardinal
direction (N/S/E/W, or O when standing on food) of the nearest pellet. Here the agent senses
each of its k nearest pellets within sensing_range as a row of
    [proximity, cos(bearing), sin(bearing), in_reach]
 - proximity = 1 - d / sensing_range, so 1 at the agent, 0 at the edge of the range
   (the same encoding as wef's sector sensors)
 - bearing: egocentric angle to the pellet (0 = straight ahead, positive = to the left),
   given as cos/sin so it is continuous across +-pi (directly behind)
 - in_reach: 1 if an EAT now could take this pellet (feeding.food_in_reach), like GridWorld's O
Rows for pellets that are not sensed (fewer than k within range) are all zeros, i.e. "nothing
detected"; there is no separate flag saying whether food exists.

Like feeding.py, these are pure functions: they take the pose and the food positions as
arguments and do not change anything.
"""

import numpy as np

from feeding import food_in_reach
from movement import wrap_angle

FOOD_FEATURES = 4   # columns per sensed pellet: proximity, cos(bearing), sin(bearing), in_reach


def egocentric_bearing(position, heading, points):
    """
    Angle (rad, in [-pi, pi]) of each point as seen by the agent, relative to its heading:
    0 = straight ahead, positive = to the left (counterclockwise), +-pi = directly behind.
    points is an (N, 2) array; returns an (N,) array.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    delta = points - np.asarray(position, dtype=float)
    # arctan2 gives the angle of the vector from position to each point, in [-pi, pi], relative to the x-axis.
    # by doing it on delta we make it relative to current agent position, and then we subtract heading to make it egocentric (relative to the agent's heading).
    return wrap_angle(np.arctan2(delta[:, 1], delta[:, 0]) - heading)


def nearest_food_observation(position, heading, food_positions, k, sensing_range, eat_radius,
                             mouth_angle=None):
    """
    (k, FOOD_FEATURES) array describing the k nearest pellets within sensing_range, nearest first.
    Each row is [proximity, cos(bearing), sin(bearing), in_reach]; rows beyond the number of
    pellets sensed are zeros. eat_radius and mouth_angle define in_reach, as in food_in_reach.
    """
    food_positions = np.asarray(food_positions, dtype=float).reshape(-1, 2)
    observation = np.zeros((k, FOOD_FEATURES))

    distances = np.linalg.norm(food_positions - np.asarray(position, dtype=float), axis=1)
    sensed = np.flatnonzero(distances < sensing_range) # strict, like eat_radius; a flag for "sensed" vs "not sensed"
    nearest = sensed[np.argsort(distances[sensed], kind="stable")][:k] # take the sensed food and sort by distance
    if len(nearest) == 0: # we have sensed no food
        return observation # return all zeros

    bearings = egocentric_bearing(position, heading, food_positions[nearest])
    reachable = food_in_reach(position, heading, food_positions, eat_radius, mouth_angle)
    n = len(nearest)
    observation[:n, 0] = 1.0 - distances[nearest] / sensing_range
    observation[:n, 1] = np.cos(bearings)
    observation[:n, 2] = np.sin(bearings)
    observation[:n, 3] = np.isin(nearest, reachable) # 0 for not reachable food
    return observation
