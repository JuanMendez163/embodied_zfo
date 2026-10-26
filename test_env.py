"""
Sanity checks for env.py. Run with: pytest test_env.py
"""

import math

import numpy as np
from gymnasium.utils.env_checker import check_env

from env import ZebrafishEnv

IDLE = [0.0, 0.0, 0.0]
EAT = [0.0, 0.0, 1.0]


def env_with_pellet_ahead(**kwargs):
    """
    Agent at the arena center facing +x, one pellet 1 unit straight ahead, no other food.
    """
    env = ZebrafishEnv(seed=0, **kwargs)
    env.reset(seed=0)
    env.position = np.array([50.0, 50.0])
    env.heading = 0.0
    env.arena.food_positions = np.array([[51.0, 50.0]])
    return env


def test_passes_gymnasium_checker():
    for kwargs in [{}, {"obs_mode": "internal"}, {"n_nearest": 3}, {"sensing_range": 20.0}]:
        check_env(ZebrafishEnv(**kwargs), skip_render_check=True)


def test_observation_layout():
    env = env_with_pellet_ahead()
    obs = env._get_obs()
    assert obs.shape == (6,) and env.observation_space.contains(obs)
    glucose, can_eat, proximity, cos_b, sin_b, in_reach = obs
    assert math.isclose(glucose, env.glucose_target / env.glucose_max)
    assert can_eat == 1.0 and in_reach == 1.0
    assert math.isclose(proximity, 1 - 1 / math.hypot(100.0, 100.0), rel_tol=1e-6)
    assert math.isclose(cos_b, 1.0) and abs(sin_b) < 1e-6


def test_observation_after_eating_last_pellet_senses_nothing():
    env = env_with_pellet_ahead()
    obs = env.step(EAT)[0]
    assert obs[1] == 0.0                   # cooling down
    assert not obs[2:].any()               # no food left -> nothing detected


def test_internal_obs_mode_is_blind():
    env = ZebrafishEnv(obs_mode="internal")
    assert env.reset(seed=0)[0].shape == (2,)


def test_reset_is_reproducible():
    a, b = ZebrafishEnv(), ZebrafishEnv()
    a.reset(seed=3)
    b.reset(seed=3)
    assert np.array_equal(a.position, b.position) and a.heading == b.heading
    assert np.array_equal(a.arena.food_positions, b.arena.food_positions)


def test_reset_state():
    env = ZebrafishEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["glucose"] == env.glucose_target and info["food_left"] == 20 and info["total_eaten"] == 0
    assert env.can_eat()


def test_idle_step_decays_glucose_and_is_penalized_at_target():
    env = ZebrafishEnv()
    env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(IDLE)
    assert info["glucose"] < env.glucose_target
    assert reward < 0                      # drifted away from the target
    assert not terminated and not truncated


def test_actions_are_clipped():
    env = ZebrafishEnv()
    env.reset(seed=0)
    start = env.position.copy()
    env.step([5.0, 0.0, 0.0])              # move_cmd clipped to 1
    assert math.isclose(np.linalg.norm(env.position - start), env.max_speed * env.dt)


def test_eat_with_pellet_in_mouth():
    env = env_with_pellet_ahead()
    _, reward, _, _, info = env.step(EAT)
    assert info["n_eaten"] == 1 and info["total_eaten"] == 1 and info["food_left"] == 0
    assert info["glucose"] > env.glucose_target
    assert not env.can_eat()


def test_eat_cmd_below_threshold_does_not_eat():
    env = env_with_pellet_ahead()
    _, _, _, _, info = env.step([0.0, 0.0, 0.4])
    assert info["n_eaten"] == 0 and info["food_left"] == 1


def test_pellet_behind_cannot_be_eaten():
    env = env_with_pellet_ahead()
    env.heading = math.pi                  # now facing away from the pellet
    _, _, _, _, info = env.step(EAT)
    assert info["n_eaten"] == 0 and info["food_left"] == 1


def test_cooldown_blocks_then_allows_eating():
    env = env_with_pellet_ahead(eat_cooldown=0.5)
    env.arena.food_positions = np.array([[51.0, 50.0], [51.5, 50.0]])
    assert env.step(EAT)[4]["n_eaten"] == 1
    for _ in range(4):                     # 0.4 s later: still cooling down
        assert env.step(EAT)[4]["n_eaten"] == 0
    assert env.step(EAT)[4]["n_eaten"] == 1  # 0.5 s after the first eat


def test_cooldown_is_in_seconds_not_steps():
    # the same 0.5 s cooldown takes twice as many steps at half the dt
    for dt, steps_blocked in [(0.1, 4), (0.05, 9)]:
        env = env_with_pellet_ahead(dt=dt, eat_cooldown=0.5)
        env.arena.food_positions = np.array([[51.0, 50.0], [51.5, 50.0]])
        env.step(EAT)
        blocked = 0
        while env.step(EAT)[4]["n_eaten"] == 0:
            blocked += 1
        assert blocked == steps_blocked


def test_starvation_terminates():
    env = ZebrafishEnv()
    env.reset(seed=0)
    env.glucose = 1.0                      # just above the threshold; one idle step pushes it below
    _, _, terminated, truncated, info = env.step(IDLE)
    assert terminated and not truncated and info["end_reason"] == "starved"


def test_time_limit_truncates():
    env = ZebrafishEnv(episode_duration=1.0, dt=0.1)
    env.reset(seed=0)
    for _ in range(9):
        assert not env.step(IDLE)[3]
    _, _, terminated, truncated, info = env.step(IDLE)
    assert truncated and not terminated and info["end_reason"] == "time limit"
