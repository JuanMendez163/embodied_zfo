"""
Sanity checks for training/metrics.py, on hand-built step infos. Run with: pytest test_metrics.py
"""

import math

import numpy as np

from training.metrics import episode_metrics, stack_episodes, summarize

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


def test_opportunities_do_not_depend_on_dt():
    # the same 3 s bout sampled at two dts
    at_dt = [episode_metrics(make_infos([True] * round(3.0 / dt)), dt, TARGET, 3.0, COOLDOWN)
             for dt in (0.1, 0.05)]
    assert at_dt[0]["eat_opportunities"] == at_dt[1]["eat_opportunities"] == 2


def test_summarize_skips_labels_and_undefined_episodes():
    per_episode = stack_episodes([
        {"timestep": 100, "env": 0, "eat_efficiency": 0.5},
        {"timestep": 100, "env": 1, "eat_efficiency": np.nan},   # no EAT presses
        {"timestep": 200, "env": 0, "eat_efficiency": 1.0},
    ])
    summary = summarize(per_episode)
    assert set(summary) == {"eat_efficiency"}
    assert math.isclose(summary["eat_efficiency"], 0.75)
