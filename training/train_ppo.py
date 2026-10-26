"""
Train a PPO actor-critic (Stable-Baselines3) to keep glucose near the target, then compare it with a
random policy and the scripted forager (zfo.policies.forage_action) on the same evaluation episodes
(training/metrics.py).
Run with: python training/train_ppo.py [--timesteps N] [--name NAME]  (writes runs/NAME/: model, progress.csv)

Choices for this first run (see docs/PORTING_NOTES.md):
 - gamma from a horizon in seconds, exp(-dt / HORIZON), so the planning horizon does not change with dt
 - glucose_start sampled in GLUCOSE_START at every reset, so the agent also experiences being sated
 - actions rescaled to [-1, 1] (RescaleAction): PPO's Gaussian starts at mean 0, std 1, which maps to
   the middle of each command's range instead of half the samples clipping at move_cmd = 0
 - no VecNormalize: observations are already in [-1, 1] and the reward is scaled in the env
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from training.metrics import episode_metrics, evaluate, ppo_policy, print_comparison
from zfo.env import ZebrafishEnv
from zfo.policies import forage_action, random_policy

DT = 0.1                          # s per step
HORIZON = 30.0                    # s; discounting time constant (~ the glucose half-life). This represents the agent's planning range. TODO can be informed by literature
GAMMA = math.exp(-DT / HORIZON)   # ~0.9967 at dt = 0.1; gamma is the actual discount factor, and we make it dt-independent by deriving it from a time constant. The agent sees the same planning horizon at any dt.
GLUCOSE_START = (20.0, 80.0)      # range for the starting glucose
N_ENVS = 8                        # parallel copies of the env
N_STEPS = 1024                    # steps per env per rollout (8192 per PPO update)
TOTAL_TIMESTEPS = 1_000_000       # ~28 simulated hours
SEED = 0
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"   # repo-root runs/, wherever this is run from


def make_env(seed=None):
    """
    The env in physical action units, as the random and scripted baselines use it.
    """
    return ZebrafishEnv(dt=DT, glucose_start=GLUCOSE_START, seed=seed)


def make_ppo_env(seed=None):
    """
    The env as PPO sees it, in training and evaluation alike: actions in [-1, 1], mapped by
    RescaleAction to the physical ranges. The only place that translation is defined.
    """
    return RescaleAction(make_env(seed), min_action=-1.0, max_action=1.0)


class HomeostasisCallback(BaseCallback):
    """
    Logs the mean episode_metrics of the training episodes that ended in each rollout, under
    "homeostasis/" in SB3's logger (printed and written to progress.csv).
    """

    def __init__(self, dt, glucose_target, episode_duration, eat_cooldown):
        super().__init__()
        self.dt = dt
        self.glucose_target = glucose_target
        self.episode_duration = episode_duration
        self.eat_cooldown = eat_cooldown

    def _on_training_start(self):
        self.traces = [[] for _ in range(self.training_env.num_envs)]   # infos so far, per env
        self.finished = []                                              # metrics of ended episodes

    def _on_step(self):
        for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
            self.traces[i].append(info)
            if done:    # the VecEnv has already reset this env, but info is still the final step's
                self.finished.append(episode_metrics(self.traces[i], self.dt, self.glucose_target,
                                                     self.episode_duration, self.eat_cooldown))
                self.traces[i] = []
        return True

    def _on_rollout_end(self):
        for key in (self.finished[0] if self.finished else {}):
            self.logger.record(f"homeostasis/{key}", np.mean([m[key] for m in self.finished]))
        self.finished = []


if __name__ == "__main__":

    # Reads command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--name", default="ppo")
    args = parser.parse_args()
    run_dir = os.path.join(RUNS_DIR, args.name)

    vec_env = make_vec_env(make_ppo_env, n_envs=N_ENVS, seed=SEED) # creates N_ENVS copies of the env, each with a different seed
    model = PPO("MlpPolicy", vec_env, n_steps=N_STEPS, batch_size=256, gamma=GAMMA, gae_lambda=0.95,
                seed=SEED, device="cpu", verbose=0)   # small MLPs: CPU is as fast as MPS for PPO
    model.set_logger(configure(run_dir, ["stdout", "csv"]))
    probe = make_env()
    # callback: runs at every env step and collects glucose traces
    # at the end of each rollout it logs mean episode_metrics of the episodes that finished
    callback = HomeostasisCallback(probe.dt, probe.glucose_target, probe.max_steps * probe.dt,
                                   probe.eat_cooldown)
    model.learn(args.timesteps, callback=callback)
    model.save(os.path.join(run_dir, "model"))

    print_comparison({
        "random": evaluate(random_policy(SEED), make_env),
        "scripted": evaluate(lambda env, obs: forage_action(env), make_env),
        "ppo": evaluate(ppo_policy(model), make_ppo_env),
    })
