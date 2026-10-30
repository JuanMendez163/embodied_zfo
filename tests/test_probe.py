"""
Sanity checks for training/probe.py. Run with: pytest test_probe.py
"""

import math

import numpy as np
from stable_baselines3 import PPO

from training.probe import eat_probability, policy_gaussian, probe_observations
from training.train_ppo import make_ppo_env


def test_probe_observation_matches_env():
    # agent facing +x with one pellet 10 units ahead and 45 degrees to the left
    env = make_ppo_env()
    env.reset(seed=0)
    base = env.unwrapped
    base.position, base.heading = np.array([50.0, 50.0]), 0.0
    base.arena.food_positions = 50.0 + 10.0 * np.array([[math.cos(math.pi / 4), math.sin(math.pi / 4)]])
    expected = base._get_obs()
    probe = probe_observations(env, base.glucose, can_eat=float(base.can_eat()),
                               proximity=1 - 10.0 / base.sensing_range, bearing=math.pi / 4, in_reach=0.0)
    assert probe.shape == (1, 6)
    assert np.allclose(probe[0], expected, atol=1e-6)


def test_probe_observations_broadcast_a_sweep():
    env = make_ppo_env()
    obs = probe_observations(env, np.linspace(0, 100, 7), proximity=0.5)
    assert obs.shape == (7, 6) and np.allclose(obs[:, 2], 0.5)


def test_policy_mean_is_the_deterministic_action():
    # an untrained model: clip(mean) mapped to physical units is what predict + RescaleAction apply
    env = make_ppo_env()
    model = PPO("MlpPolicy", env, seed=0, device="cpu")
    obs = probe_observations(env, np.linspace(10, 90, 5), proximity=0.6, bearing=0.3)
    mean, std = policy_gaussian(model, env, obs)
    low, high = env.unwrapped.action_space.low, env.unwrapped.action_space.high
    applied = np.array([env.action(model.predict(o, deterministic=True)[0]) for o in obs])
    assert np.allclose(np.clip(mean, low, high), applied, atol=1e-5)
    assert np.allclose(std, (high - low) / 2)          # log_std starts at 0: std 1 in [-1, 1] units


def test_eat_probability():
    assert math.isclose(eat_probability(0.5, 0.2), 0.5)
    assert eat_probability(0.9, 0.1) > 0.99 and eat_probability(0.1, 0.1) < 0.01
