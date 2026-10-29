"""
Sanity checks for training/train_ppo.py. Run with: pytest test_train_ppo.py
"""

import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from training.train_ppo import make_ppo_env
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
