"""
Sanity checks for feeding.py. Run with: pytest test_feeding.py
"""

import math

import numpy as np

from zfo.feeding import food_in_reach, nearest_food_in_reach

AGENT = np.array([50.0, 50.0])
EAT_RADIUS = 2.0


def test_no_food():
    assert food_in_reach(AGENT, 0.0, np.zeros((0, 2)), EAT_RADIUS).size == 0
    assert nearest_food_in_reach(AGENT, 0.0, np.zeros((0, 2)), EAT_RADIUS) is None


def test_only_pellets_within_radius():
    food = np.array([[51.0, 50.0],    # 1 away: in reach
                     [53.0, 50.0],    # 3 away: out of reach
                     [50.0, 48.5]])   # 1.5 away: in reach
    assert set(food_in_reach(AGENT, 0.0, food, EAT_RADIUS)) == {0, 2}


def test_radius_is_strict():
    food = np.array([[52.0, 50.0]])   # exactly eat_radius away
    assert food_in_reach(AGENT, 0.0, food, EAT_RADIUS).size == 0


def test_nearest_first():
    food = np.array([[51.5, 50.0], [50.5, 50.0], [51.0, 50.0]])
    assert list(food_in_reach(AGENT, 0.0, food, EAT_RADIUS)) == [1, 2, 0]
    assert nearest_food_in_reach(AGENT, 0.0, food, EAT_RADIUS) == 1


def test_without_mouth_angle_heading_is_ignored():
    behind = np.array([[49.0, 50.0]])
    assert nearest_food_in_reach(AGENT, 0.0, behind, EAT_RADIUS) == 0


def test_mouth_angle_keeps_food_in_front():
    food = np.array([[51.0, 50.0],    # straight ahead (heading 0)
                     [49.0, 50.0],    # directly behind
                     [50.0, 51.0]])   # 90° to the left
    in_reach = food_in_reach(AGENT, 0.0, food, EAT_RADIUS, mouth_angle=math.radians(90))
    assert list(in_reach) == [0]


def test_mouth_angle_follows_heading():
    left = np.array([[50.0, 51.0]])   # 90° to the left of +x
    assert nearest_food_in_reach(AGENT, math.pi / 2, left, EAT_RADIUS, mouth_angle=math.radians(60)) == 0
    assert nearest_food_in_reach(AGENT, 0.0, left, EAT_RADIUS, mouth_angle=math.radians(60)) is None


def test_mouth_angle_across_the_pi_boundary():
    # heading ~ +179°, pellet at ~ -179°: only ~2° apart, so it must count as in front
    heading = math.radians(179)
    pellet = AGENT + 1.0 * np.array([math.cos(math.radians(-179)), math.sin(math.radians(-179))])
    assert nearest_food_in_reach(AGENT, heading, pellet[None, :], EAT_RADIUS, mouth_angle=math.radians(30)) == 0


def test_mouth_angle_edges():
    # cone of 90° -> pellets at ±40° are in, at ±50° are out
    angles = np.radians([40, -40, 50, -50])
    food = AGENT + np.stack([np.cos(angles), np.sin(angles)], axis=1)
    assert set(food_in_reach(AGENT, 0.0, food, EAT_RADIUS, mouth_angle=math.radians(90))) == {0, 1}


def test_does_not_mutate_inputs():
    food = np.array([[51.0, 50.0], [49.0, 50.0]])
    before = food.copy()
    food_in_reach(AGENT, 0.0, food, EAT_RADIUS, mouth_angle=math.radians(90))
    assert np.array_equal(food, before)
