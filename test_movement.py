"""
Sanity checks for movement.py. Run with: pytest test_movement.py
"""

import math

import numpy as np

from movement import enforce_bounds, propose_pose, step_pose, wrap_angle

ARENA = (100.0, 100.0)
CENTER = np.array([50.0, 50.0])
MAX_SPEED = 10.0     # units/s
MAX_TURN = math.pi   # rad/s
DT = 0.1             # s


def test_wrap_angle():
    assert math.isclose(wrap_angle(3 * math.pi / 2), -math.pi / 2)
    assert math.isclose(wrap_angle(-3 * math.pi / 2), math.pi / 2)
    assert math.isclose(wrap_angle(0.5), 0.5)


def test_idle_does_not_move():
    pos, heading = propose_pose(CENTER, 0.3, 0.0, 0.0, MAX_SPEED, MAX_TURN, DT)
    assert np.allclose(pos, CENTER)
    assert math.isclose(heading, 0.3)


def test_straight_line_along_heading():
    # full forward along +y for one step covers max_speed * dt
    pos, heading = propose_pose(CENTER, math.pi / 2, 1.0, 0.0, MAX_SPEED, MAX_TURN, DT)
    assert np.allclose(pos, CENTER + [0.0, MAX_SPEED * DT])
    assert math.isclose(heading, math.pi / 2)


def test_turn_in_place():
    pos, heading = propose_pose(CENTER, 0.0, 0.0, 1.0, MAX_SPEED, MAX_TURN, DT)
    assert np.allclose(pos, CENTER)
    assert math.isclose(heading, MAX_TURN * DT)


def test_full_circle_returns_near_start():
    # constant move + turn traces a closed polygon approximating a circle
    n_steps = round(2 * math.pi / (MAX_TURN * DT))
    pos, heading = CENTER.copy(), 0.0
    for _ in range(n_steps):
        pos, heading, hit_wall = step_pose(pos, heading, 1.0, 1.0, MAX_SPEED, MAX_TURN, DT, ARENA)
        assert not hit_wall
    assert np.allclose(pos, CENTER, atol=1e-9)


def test_does_not_mutate_input():
    start = CENTER.copy()
    propose_pose(start, 0.0, 1.0, 0.0, MAX_SPEED, MAX_TURN, DT)
    assert np.array_equal(start, CENTER)


def test_wall_clips_and_reports_hit():
    near_wall = np.array([99.5, 50.0])
    pos, heading, hit_wall = step_pose(near_wall, 0.0, 1.0, 0.0, MAX_SPEED, MAX_TURN, DT,
                                       ARENA, body_radius=1.0)
    assert hit_wall
    assert np.allclose(pos, [99.0, 50.0])


def test_enforce_bounds_inside_is_unchanged():
    pos, hit_wall = enforce_bounds(CENTER, ARENA, body_radius=1.0)
    assert not hit_wall
    assert np.allclose(pos, CENTER)
