"""
Sanity checks for training/train_ppo.py. Run with: pytest test_train_ppo.py
"""

import numpy as np
from stable_baselines3.common.env_util import make_vec_env

import pytest

from training.train_ppo import make_ppo_env, run_env_kwargs
from zfo.metabolism import drive_reward


def test_ppo_reward_is_scaled_drive_reward():
    # the reward PPO receives, through RescaleAction, Monitor and the vec env, is the env's
    # drive_reward / reward_scale, unchanged by any wrapper
    vec_env = make_vec_env(make_ppo_env, n_envs=2, seed=0)
    vec_env.reset()
    rng = np.random.default_rng(0)
    target, n, m, scale = (vec_env.get_attr(name)[0]
                           for name in ["glucose_target", "drive_n", "drive_m", "reward_scale"])
    for _ in range(200):
        glucose_before = np.array(vec_env.get_attr("glucose"))
        actions = rng.uniform(-1.0, 1.0, size=(2, 3))
        _, rewards, _, infos = vec_env.step(actions)
        glucose_after = np.array([info["glucose"] for info in infos])
        expected = drive_reward(glucose_before, glucose_after, target, n, m) / scale
        assert np.allclose(rewards, expected)


def test_default_run_hides_flags_with_attempt_cost():
    kwargs = run_env_kwargs()
    assert kwargs == {"observe_can_eat": False, "observe_in_reach": False,
                      "eat_attempt_resets_cooldown": True, "eat_press_resets_cooldown": False,
                      "food_regrowth_interval": 10.0, "food_per_regrowth": 1, "episode_duration": 300.0}
    env = make_ppo_env(**kwargs)
    assert env.observation_space.shape == (4,)      # glucose, proximity, cos, sin
    assert env.unwrapped.max_steps == round(300.0 / env.unwrapped.dt)


def test_run_env_kwargs_options():
    assert run_env_kwargs(hide=[], eat_cost="none", food_regrowth_interval=0, episode_duration=120.0,
                          food_per_regrowth=3) == {
        "observe_can_eat": True, "observe_in_reach": True,
        "eat_attempt_resets_cooldown": False, "eat_press_resets_cooldown": False,
        "food_regrowth_interval": None, "food_per_regrowth": 3, "episode_duration": 120.0}
    assert run_env_kwargs(hide=["in_reach"], eat_cost="press")["eat_press_resets_cooldown"]
    with pytest.raises(ValueError):
        run_env_kwargs(eat_cost="free")
