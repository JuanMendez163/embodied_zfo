"""
Sanity checks for training/metrics.py, on hand-built step infos. Run with: pytest test_metrics.py
"""

import math

import numpy as np

from training.metrics import episode_metrics, stack_episodes, summarize
from zfo.env import ZebrafishEnv

DT = 0.1
TARGET = 50.0
COOLDOWN = 2.0


def make_infos(eat, reach=None, ready=None, eaten=None, glucose=TARGET, end_reason="time limit"):
    """
    Step infos for an episode of len(eat) steps. eat, reach, ready, eaten: per-step booleans
    (reach and ready default to always True, eaten to never). Glucose is constant.
    """
    n = len(eat)
    reach = [True] * n if reach is None else reach
    ready = [True] * n if ready is None else ready
    eaten = [False] * n if eaten is None else eaten
    infos = [{"glucose": glucose, "action": np.array([0.0, 0.0, float(e)]), "eat_attempted": e,
              "eat_ready": r, "food_in_reach": f, "n_eaten": int(x)}
             for e, f, r, x in zip(eat, reach, ready, eaten)]
    infos[-1].update({"total_eaten": sum(eaten), "end_reason": end_reason, "time": n * DT})
    return infos


def metrics(infos, duration=None):
    """
    episode_metrics with this file's constants; the episode duration defaults to the episode length.
    """
    duration = len(infos) * DT if duration is None else duration
    return episode_metrics(infos, DT, TARGET, duration, COOLDOWN)


def test_single_tap_that_eats_is_fully_efficient():
    eat = [False] * 5 + [True] + [False] * 5
    m = metrics(make_infos(eat, eaten=eat))
    assert m["eat_opportunities"] == 1
    assert m["eat_efficiency"] == 1.0
    assert math.isclose(m["eat_held_time"], DT)


def test_bout_of_one_cooldown_counts_one_opportunity():
    n_hold = round(COOLDOWN / DT)   # held for exactly one cooldown
    m = metrics(make_infos([True] * n_hold + [False] * 3))
    assert m["eat_opportunities"] == 1


def test_no_eat_presses_gives_nan_ratios():
    m = metrics(make_infos([False] * 10))
    for key in ("eat_efficiency", "eat_out_of_reach", "eat_during_cooldown"):
        assert np.isnan(m[key])
    assert m["eat_opportunities"] == 0 and m["eat_held_time"] == 0


def test_failure_fractions_split_eat_held_time():
    eat = [True] * 4
    reach = [False, True, True, True]
    ready = [True, False, False, True]
    m = metrics(make_infos(eat, reach=reach, ready=ready))
    assert math.isclose(m["eat_out_of_reach"], 0.25)
    assert math.isclose(m["eat_during_cooldown"], 0.5)


def test_starved_time_counts_as_target_error():
    # alive half the duration exactly at target, then starved: mean error = target / 2
    infos = make_infos([False] * 10, end_reason="starved")
    m = metrics(infos, duration=2 * len(infos) * DT)
    assert m["starved"] == 1.0
    assert math.isclose(m["glucose_target_error"], TARGET / 2)


def test_in_range_time_counts_only_alive_time_inside():
    # alive 1 s inside the range, then starved for the rest of a 2 s episode: 1 s in range
    infos = make_infos([False] * 10, end_reason="starved")
    m = episode_metrics(infos, DT, TARGET, 2 * len(infos) * DT, COOLDOWN, glucose_range=(40.0, 60.0))
    assert math.isclose(m["in_range_time"], len(infos) * DT)
    m = episode_metrics(infos, DT, TARGET, len(infos) * DT, COOLDOWN, glucose_range=(55.0, 60.0))
    assert m["in_range_time"] == 0.0                 # glucose 50 is below this range


def test_opportunities_do_not_depend_on_dt():
    # the same 3 s bout sampled at two dts
    at_dt = [episode_metrics(make_infos([True] * round(3.0 / dt)), dt, TARGET, 3.0, COOLDOWN)
             for dt in (0.1, 0.05)]
    assert at_dt[0]["eat_opportunities"] == at_dt[1]["eat_opportunities"] == 2


def test_strikes_do_not_depend_on_dt():
    # a 3 s hold, ready throughout: strikes at t = 0 and t = COOLDOWN, at either dt
    at_dt = [episode_metrics(make_infos([True] * round(3.0 / dt)), dt, TARGET, 3.0, COOLDOWN)
             for dt in (0.1, 0.05)]
    assert at_dt[0]["eat_strikes"] == at_dt[1]["eat_strikes"] == 2


def test_hold_during_cooldown_strikes_once_ready():
    # held 5 steps, cooldown passes at step 2: one strike there, then none until COOLDOWN later
    m = metrics(make_infos([True] * 5, ready=[False, False, True, True, True]))
    assert m["eat_strikes"] == 1
    assert math.isclose(m["eat_strike_rate"], 1 / (5 * DT))


def test_strike_that_eats_does_not_miss():
    eat = [False] * 3 + [True] + [False] * 3
    m = metrics(make_infos(eat, eaten=eat))
    assert m["eat_strikes"] == 1 and m["eat_strike_miss"] == 0.0


def test_eat_between_counted_strikes_is_a_strike():
    # no cost: a failed strike at t = 0 leaves the env ready, so a held EAT eats at step 3, inside the
    # counter's cooldown; that eat is a strike and restarts the count (next strike COOLDOWN after it)
    n = round(2 * COOLDOWN / DT)
    eaten = [i == 3 for i in range(n)]
    m = metrics(make_infos([True] * n, eaten=eaten))
    strike_steps = [0, 3, 3 + round(COOLDOWN / DT)]
    assert m["eat_strikes"] == len(strike_steps)
    assert math.isclose(m["eat_strike_miss"], 1 - 1 / 3)


def test_eat_on_re_press_inside_counted_cooldown_is_a_strike():
    # no cost: tap at t = 0 (miss), release, tap again at step 2 and eat; the counter's clock alone
    # would skip the second tap (2 steps < COOLDOWN), but an eat always counts
    eat = [True, False, True] + [False] * 5
    eaten = [False, False, True] + [False] * 5
    m = metrics(make_infos(eat, eaten=eaten))
    assert m["eat_strikes"] == 2 and m["eat_strike_miss"] == 0.5


def test_no_strikes_gives_nan_miss_fraction():
    m = metrics(make_infos([True] * 4, ready=[False] * 4))   # held only during the cooldown
    assert m["eat_strikes"] == 0 and m["eat_strike_rate"] == 0.0
    assert np.isnan(m["eat_strike_miss"])


def test_strikes_are_ready_presses_when_attempts_restart_cooldown():
    # in the env with the EAT cost every ready press restarts the cooldown, so the counted strikes
    # are exactly the presses made with the cooldown passed
    env = ZebrafishEnv(eat_attempt_resets_cooldown=True, seed=0)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    infos = [env.step([rng.uniform(), rng.uniform(-1, 1), rng.uniform()])[4] for _ in range(300)]
    ready_presses = sum(i["eat_attempted"] and i["eat_ready"] for i in infos)
    m = episode_metrics(infos, env.dt, env.glucose_target, len(infos) * env.dt, env.eat_cooldown)
    assert ready_presses > 10
    assert m["eat_strikes"] == ready_presses


def test_strikes_are_ready_presses_when_presses_restart_cooldown():
    # with the stricter cost a ready step has had no press for a cooldown, so again every ready press
    # is a counted strike; a sparse presser (p = 0.15) so ready presses are not rare
    env = ZebrafishEnv(eat_press_resets_cooldown=True, seed=0)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    infos = [env.step([rng.uniform(), rng.uniform(-1, 1), float(rng.uniform() < 0.15)])[4] for _ in range(600)]
    ready_presses = sum(i["eat_attempted"] and i["eat_ready"] for i in infos)
    m = episode_metrics(infos, env.dt, env.glucose_target, len(infos) * env.dt, env.eat_cooldown)
    assert ready_presses > 10
    assert m["eat_strikes"] == ready_presses


def test_summarize_skips_labels_and_undefined_episodes():
    per_episode = stack_episodes([
        {"timestep": 100, "env": 0, "eat_efficiency": 0.5},
        {"timestep": 100, "env": 1, "eat_efficiency": np.nan},   # no EAT presses
        {"timestep": 200, "env": 0, "eat_efficiency": 1.0},
    ])
    summary = summarize(per_episode)
    assert set(summary) == {"eat_efficiency"}
    assert math.isclose(summary["eat_efficiency"], 0.75)
