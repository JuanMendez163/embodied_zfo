"""
Sanity checks for sensing.py. Run with: pytest test_sensing.py
"""

import math

import numpy as np

from zfo.feeding import food_in_reach
from zfo.sensing import FOOD_FEATURES, egocentric_bearing, nearest_food_observation

AGENT = np.array([50.0, 50.0])
EAT_RADIUS = 2.0
MOUTH = math.radians(90)
RANGE = 100.0


def sense(food, heading=0.0, k=1, sensing_range=RANGE):
    return nearest_food_observation(AGENT, heading, np.asarray(food, dtype=float), k, sensing_range,
                                    EAT_RADIUS, MOUTH)


# --- egocentric_bearing ---

def test_bearing_ahead_left_right_behind():
    points = AGENT + np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0]])
    bearings = egocentric_bearing(AGENT, 0.0, points)
    assert np.allclose(bearings[:3], [0.0, math.pi / 2, -math.pi / 2])
    assert math.isclose(abs(bearings[3]), math.pi)


def test_bearing_follows_heading():
    left_of_north = AGENT + np.array([[-1.0, 0.0]])        # world angle pi; heading +y (pi/2) -> 90° left
    assert np.allclose(egocentric_bearing(AGENT, math.pi / 2, left_of_north), [math.pi / 2])


def test_bearing_is_wrapped():
    # heading ~ +179°, point at ~ -179°: 2° to the left, not -358°
    point = AGENT + np.array([[math.cos(math.radians(-179)), math.sin(math.radians(-179))]])
    assert np.allclose(egocentric_bearing(AGENT, math.radians(179), point), [math.radians(2)])


def test_bearing_does_not_depend_on_distance():
    points = AGENT + np.array([[1.0, 1.0], [10.0, 10.0]])
    bearings = egocentric_bearing(AGENT, 0.0, points)
    assert np.allclose(bearings, [math.pi / 4, math.pi / 4])


# --- nearest_food_observation ---

def test_no_food_is_all_zeros():
    obs = sense(np.zeros((0, 2)), k=3)
    assert obs.shape == (3, FOOD_FEATURES) and not obs.any()


def test_pellet_ahead():
    proximity, cos_b, sin_b, in_reach = sense([[51.0, 50.0]])[0]
    assert math.isclose(proximity, 1 - 1 / RANGE)
    assert math.isclose(cos_b, 1.0) and math.isclose(sin_b, 0.0, abs_tol=1e-12)
    assert in_reach == 1.0


def test_pellet_to_the_left_out_of_reach():
    proximity, cos_b, sin_b, in_reach = sense([[50.0, 60.0]])[0]
    assert math.isclose(proximity, 0.9)
    assert math.isclose(cos_b, 0.0, abs_tol=1e-12) and math.isclose(sin_b, 1.0)
    assert in_reach == 0.0


def test_close_pellet_behind_is_not_in_reach():
    row = sense([[49.0, 50.0]])[0]
    assert math.isclose(row[1], -1.0) and row[3] == 0.0


def test_out_of_range_is_not_sensed():
    assert not sense([[80.0, 50.0]], sensing_range=20.0).any()
    assert sense([[60.0, 50.0]], sensing_range=20.0)[0, 0] > 0


def test_nearest_first_and_padded():
    food = [[70.0, 50.0], [55.0, 50.0], [50.0, 60.0]]      # distances 20, 5, 10
    obs = sense(food, k=4)
    assert np.allclose(obs[:3, 0], 1 - np.array([5.0, 10.0, 20.0]) / RANGE)
    assert not obs[3].any()


def test_in_reach_matches_feeding():
    rng = np.random.default_rng(0)
    food = AGENT + rng.uniform(-3, 3, size=(30, 2))
    heading = 0.7
    obs = nearest_food_observation(AGENT, heading, food, 30, RANGE, EAT_RADIUS, MOUTH)
    assert int(obs[:, 3].sum()) == len(food_in_reach(AGENT, heading, food, EAT_RADIUS, MOUTH))


def test_does_not_mutate_inputs():
    food = np.array([[51.0, 50.0], [49.0, 50.0]])
    before = food.copy()
    sense(food, k=2)
    assert np.array_equal(food, before)
