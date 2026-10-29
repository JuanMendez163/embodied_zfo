"""
Train a PPO actor-critic (Stable-Baselines3) to keep glucose near the target, then compare it with a
random policy and the scripted forager (zfo.policies.forage_action) on the same evaluation episodes
(training/metrics.py).
Run with: python training/train_ppo.py [--timesteps N] [--name NAME]  (writes runs/NAME/: model, progress.csv,
train_episodes.npz, eval.npz)

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

from training.metrics import (episode_metrics, evaluate, ppo_policy, print_comparison, stack_episodes,
                              summarize)
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
    Logs the summarized episode_metrics of the training episodes that ended in each rollout, under
    "homeostasis/" in SB3's logger (printed and written to progress.csv).
    Also keeps every training episode, one row each: "timestep" (total env steps, over all envs,
    when it ended), "env" (which parallel env played it) and its episode_metrics. Saved to
    save_path (.npz, {key: array (n_episodes,)}) after every rollout, so a crashed run keeps them.
    """

    # stores the physical parameters of the env that episode_metrics needs, and where to save episodes
    def __init__(self, dt, glucose_target, episode_duration, eat_cooldown, save_path):
        super().__init__()
        self.dt = dt
        self.glucose_target = glucose_target
        self.episode_duration = episode_duration
        self.eat_cooldown = eat_cooldown
        self.save_path = save_path

    def _on_training_start(self):
        self.traces = [[] for _ in range(self.training_env.num_envs)]   # infos so far, per env
        self.finished = []                                              # metrics of episodes ended this rollout
        self.episodes = [] # every ended episode, labelled. this is for storing raw episode metrics across every episode

    def _on_step(self): # runs after each vectorized step, i.e. after each step in each env
        for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
            self.traces[i].append(info) # stores info for each env i
            if done:    # the VecEnv has already reset this env, but info is still the final step's. VecEnv automatically resets env after episode finish
                metrics = episode_metrics(self.traces[i], self.dt, self.glucose_target,
                                          self.episode_duration, self.eat_cooldown)
                self.finished.append(metrics)
                # num_timesteps already counts this step (SB3 adds n_envs before calling the callback)
                self.episodes.append({"timestep": self.num_timesteps, "env": i, **metrics})
                self.traces[i] = [] # after logging the finished episode in self.finished, we wipe clean for the next env
        return True

    # runs after each rollout (N_STEPS per env): logs the summary of the episodes finished in that
    # rollout, and re-saves every episode so far
    def _on_rollout_end(self):
        if not self.finished:   # no episode ended in this rollout: nothing to log
            return
        for key, value in summarize(stack_episodes(self.finished)).items():
            self.logger.record(f"homeostasis/{key}", value)
        self.finished = []
        np.savez(self.save_path, **stack_episodes(self.episodes))


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
    # callback: runs at every env step and collects step info
    # keeps the metrics of every episode (saved to train_episodes.npz each rollout)
    # at the end of each rollout it logs the summarized episode_metrics of the episodes that finished
    callback = HomeostasisCallback(probe.dt, probe.glucose_target, probe.max_steps * probe.dt,
                                   probe.eat_cooldown, os.path.join(run_dir, "train_episodes.npz"))
    model.learn(args.timesteps, callback=callback) # does the whole training loop, until args.timesteps env steps have been collected
    model.save(os.path.join(run_dir, "model"))

    evals = {
        "random": evaluate(random_policy(SEED), make_env),
        "scripted": evaluate(lambda env, obs: forage_action(env), make_env),
        "ppo": evaluate(ppo_policy(model), make_ppo_env),
    }
    # per-episode metrics, keys "policy/metric": load with np.load(runs/NAME/eval.npz)
    np.savez(os.path.join(run_dir, "eval.npz"),
             **{f"{name}/{key}": values for name, per_episode in evals.items()
                for key, values in per_episode.items()})
    print_comparison({name: summarize(per_episode) for name, per_episode in evals.items()})
