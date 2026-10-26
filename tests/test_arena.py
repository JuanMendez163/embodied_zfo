"""
Sanity checks for the food in arena.py. Run with: pytest test_arena.py
"""

import numpy as np

from zfo.arena import Arena

START = np.array([50.0, 50.0])


def make_arena(**kwargs):
    arena = Arena(seed=0, **kwargs)
    arena.spawn_food(avoid=START)
    return arena


def test_starts_empty():
    arena = Arena(seed=0)
    assert arena.food_positions.shape == (0, 2)
    assert arena.num_food == 0


def test_spawns_n_food():
    arena = make_arena(n_food=25)
    assert arena.food_positions.shape == (25, 2)
    assert arena.num_food == 25


def test_food_respects_wall_margin():
    arena = make_arena(n_food=500, food_wall_margin=5.0)
    x, y = arena.food_positions.T
    assert x.min() >= 5.0 and x.max() <= arena.width - 5.0
    assert y.min() >= 5.0 and y.max() <= arena.height - 5.0


def test_food_avoids_agent_start():
    arena = make_arena(n_food=500, food_min_distance=15.0)
    distances = np.linalg.norm(arena.food_positions - START, axis=1)
    assert distances.min() >= 15.0


def test_no_avoid_point_allows_food_anywhere():
    arena = Arena(seed=0, n_food=2000, food_min_distance=15.0)
    arena.spawn_food()
    distances = np.linalg.norm(arena.food_positions - START, axis=1)
    assert distances.min() < 15.0


def test_same_seed_same_food():
    assert np.array_equal(make_arena().food_positions, make_arena().food_positions)


def test_spawn_replaces_existing_food():
    arena = make_arena(n_food=10)
    first = arena.food_positions.copy()
    arena.spawn_food(avoid=START)
    assert arena.num_food == 10
    assert not np.array_equal(arena.food_positions, first)


def test_remove_food():
    arena = make_arena(n_food=10)
    kept = arena.food_positions[[0, 2, 3, 4, 5, 6, 8, 9]]
    arena.remove_food([7, 1])
    assert np.array_equal(arena.food_positions, kept)


def test_remove_nothing():
    arena = make_arena(n_food=10)
    before = arena.food_positions.copy()
    arena.remove_food([])
    assert np.array_equal(arena.food_positions, before)


def test_reset_clears_food():
    arena = make_arena(n_food=10)
    arena.reset(seed=1)
    assert arena.num_food == 0
