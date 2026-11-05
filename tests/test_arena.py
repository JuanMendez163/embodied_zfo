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


# ---- regrowth (regrow_food) ----

def test_no_regrowth_draws_no_random_numbers():
    # regrowth off must leave the random stream untouched, so runs saved before regrowth replay exactly
    arena = make_arena()
    state = arena.np_random.bit_generator.state
    assert arena.regrow_food(0.1, avoid=START) == 0
    assert arena.np_random.bit_generator.state == state


def test_one_pellet_per_interval_at_any_dt():
    # ticks at t = 10, 20, ..., 120 s: exactly 12 pellets whatever dt is, each on the first step that
    # reaches its tick (the same time when dt divides the interval, at most one step late otherwise)
    for dt in (0.1, 0.02, 0.3):
        arena = Arena(seed=0, food_regrowth_interval=10.0, max_food=100)
        steps = round(120.0 / dt)
        grown = [arena.regrow_food(dt) for _ in range(steps)]
        lateness = np.array([(i + 1) * dt for i, g in enumerate(grown) if g]) - np.arange(10.0, 121.0, 10.0)
        assert sum(grown) == 12
        assert np.all((lateness > -1e-6) & (lateness < dt - 1e-6))


def test_batch_per_tick_at_any_dt():
    # food_per_regrowth = 3 with room to spare: 3 pellets at each of the 12 ticks in 120 s, whatever dt is
    for dt in (0.1, 0.3, 25.0):
        arena = Arena(seed=0, food_regrowth_interval=10.0, food_per_regrowth=3, max_food=100)
        assert sum(arena.regrow_food(dt) for _ in range(round(120.0 / dt))) == 36


def test_batch_that_does_not_fit_grows_partially():
    # 19 of 20 pellets, batch of 3: the tick grows the 1 that fits; the other 2 are not saved
    arena = make_arena(n_food=19, food_regrowth_interval=1.0, food_per_regrowth=3, max_food=20)
    assert arena.regrow_food(1.0) == 1 and arena.num_food == 20
    arena.remove_food([0, 1, 2, 3])
    assert arena.regrow_food(1.0) == 3 and arena.num_food == 19


def test_regrowth_stops_at_max_food():
    # interval shorter than dt: several ticks per step, but never more pellets than max_food
    arena = make_arena(n_food=5, food_regrowth_interval=0.04, max_food=8)
    counts = []
    for _ in range(5):
        arena.regrow_food(0.1, avoid=START)
        counts.append(arena.num_food)
    assert counts == [7, 8, 8, 8, 8]


def test_tick_at_full_arena_is_not_saved():
    arena = make_arena(n_food=3, food_regrowth_interval=1.0, max_food=3)
    assert sum(arena.regrow_food(0.5) for _ in range(10)) == 0     # full for 5 s: 5 ticks, nothing grows
    arena.remove_food([0])
    assert arena.regrow_food(0.5) == 0                              # next tick is at 5.5 s, not now
    assert arena.regrow_food(0.5) == 1 and arena.num_food == 3


def test_regrown_food_avoids_agent_and_walls():
    arena = Arena(seed=0, food_regrowth_interval=0.001, max_food=200)
    agent = np.array([20.0, 30.0])
    arena.regrow_food(0.2, avoid=agent)    # 200 ticks in one step
    assert arena.num_food == 200
    assert np.all(np.linalg.norm(arena.food_positions - agent, axis=1) >= arena.food_min_distance)
    m = arena.food_wall_margin
    assert np.all((arena.food_positions >= m) & (arena.food_positions <= np.array(arena.size) - m))


def test_reset_restarts_regrowth_clock():
    arena = Arena(seed=0, food_regrowth_interval=10.0, max_food=5)
    for _ in range(95):
        arena.regrow_food(0.1)             # 9.5 s: no tick yet
    arena.reset()
    assert sum(arena.regrow_food(0.1) for _ in range(95)) == 0
