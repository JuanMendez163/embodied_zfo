"""
Probing a trained PPO policy with hand-built observations: what the actor's Gaussian does as one
observation feature is swept (e.g. glucose or the bearing to food) while the others are held fixed.
No episodes are run, so a probe shows the learned rule itself, including in states the agent rarely
visits.

Actions are reported in the env's physical units (move_cmd in [0, 1], turn_cmd in [-1, 1], eat_cmd in
[0, 1]): the mean is mapped through the same affine map as RescaleAction (training.train_ppo.make_ppo_env)
but not clipped, so a mean beyond the action bounds stays visible. The deterministic policy (evaluation,
viz/animate_ppo.py) takes clip(mean), so it holds EAT where mean_eat > EAT_THRESHOLD.
"""

import math

import numpy as np
import torch

from zfo.env import EAT_THRESHOLD


def probe_observations(env, glucose, can_eat=1.0, proximity=0.0, bearing=0.0, in_reach=0.0):
    """
    (N, obs_dim) observations in the layout of env._get_obs (obs_mode="nearest", n_nearest=1):
    [glucose / glucose_max, can_eat, proximity, cos(bearing), sin(bearing), in_reach].
    Arguments broadcast against each other, so sweep one (an array) and hold the rest fixed (scalars).
    glucose is in env units; bearing in rad (0 = ahead, positive = left).
    Not every combination occurs in the env: in_reach = 1 needs proximity > 1 - eat_radius / sensing_range
    and |bearing| < mouth_angle / 2, and "nothing sensed" is an all-zero food row (cos and sin both 0),
    which no bearing gives; for that, set observations[:, 2:] = 0.
    """
    if env.unwrapped.obs_mode != "nearest" or env.unwrapped.n_nearest != 1:
        raise ValueError("probe_observations assumes obs_mode='nearest' and n_nearest=1")
    glucose, can_eat, proximity, bearing, in_reach = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (glucose, can_eat, proximity, bearing, in_reach)))
    columns = [glucose / env.unwrapped.glucose_max, can_eat, proximity, np.cos(bearing), np.sin(bearing),
               in_reach]
    return np.column_stack([np.ravel(c) for c in columns]).astype(np.float32)


def policy_gaussian(model, env, observations):
    """
    Mean (N, 3) and standard deviation (3,) of the actor's Gaussian for each observation, in physical
    action units (unclipped). The std is state-independent (SB3's log_std parameter).
    """
    obs_tensor, _ = model.policy.obs_to_tensor(observations)
    with torch.no_grad():
        mean = model.policy.get_distribution(obs_tensor).distribution.mean.cpu().numpy()
        std = model.policy.log_std.exp().cpu().numpy()
    low, high = env.unwrapped.action_space.low, env.unwrapped.action_space.high
    half_range = (high - low) / 2
    return low + (mean + 1.0) * half_range, std * half_range


def eat_probability(mean_eat, std_eat, threshold=EAT_THRESHOLD):
    """
    Probability that a sampled eat_cmd exceeds threshold, i.e. that the stochastic (training) policy holds
    EAT: P(N(mean, std) > threshold). Clipping to [0, 1] does not change which side of the threshold a
    sample is on.
    """
    z = (np.asarray(mean_eat, dtype=float) - threshold) / (std_eat * math.sqrt(2.0))
    return 0.5 * (1.0 + np.vectorize(math.erf)(z))
