"""
Train a PPO actor-critic (Stable-Baselines3) to keep glucose near the target, then compare it with a
random policy (a lower bound) on the same evaluation episodes (training/metrics.py). Trained agents are
compared with each other across runs (eval.npz, viz/compare_training.py).
Run with: python training/train_ppo.py [--timesteps N] [--name NAME] [--hide [FLAG ...]] [--eat-cost RULE]
                                     [--food-regrowth-interval S] [--food-per-regrowth N] [--episode-duration S]
                                     [--sensing-range R] [--reward drive|range] [--glucose-range LOW HIGH]
                                     [--starvation-penalty P]
(writes runs/NAME/: model, env_kwargs.json, progress.csv, train_episodes.npz, eval.npz)
Defaults (since 2026-11-03): can_eat and in_reach hidden, attempt cost, one pellet regrown every 10 s,
300 s episodes (HIDE_FLAGS, EAT_COST, FOOD_REGROWTH_INTERVAL, EPISODE_DURATION below).
--hide FLAG ... drops those flags from the observation (ZebrafishEnv's observe_can_eat / observe_in_reach);
  --hide with no flags observes both.
--eat-cost none | attempt | press: failed EATs are free (GridWorld) | a press with the cooldown passed
  restarts it, eaten or not (eat_attempt_resets_cooldown) | every press restarts it, even during the
  cooldown (eat_press_resets_cooldown).
--food-regrowth-interval S: one pellet grows every S seconds while the arena holds fewer than n_food
  (ZebrafishEnv's food_regrowth_interval); 0 turns regrowth off.
--food-per-regrowth N: grow N pellets at each of those ticks instead of one (food_per_regrowth), i.e.
  food arrives in batches; the mean rate is N / S pellets per second.
--episode-duration S: episode time limit in seconds (ZebrafishEnv's episode_duration; the env default is 120).
--sensing-range R: pellets farther than R (arena units) are not sensed (ZebrafishEnv's sensing_range);
  0 = the arena diagonal, i.e. sees everything (the default).
--reward drive | range: reward the reduction in distance from the glucose target (default) | +1 per second
  with glucose inside --glucose-range LOW HIGH (default 40 60), 0 outside it, and -P once on starving
  (--starvation-penalty, default 100) (ZebrafishEnv's reward_mode, glucose_range, starvation_penalty).
env_kwargs.json records every one of these settings explicitly, so a run replays the same however the
defaults change. ZebrafishEnv's own defaults are unchanged (all flags observed, no cost), so runs saved
with env_kwargs.json = {} still replay as trained.

Choices for this first run (see docs/PORTING_NOTES.md):
 - gamma from a horizon in seconds, exp(-dt / HORIZON), so the planning horizon does not change with dt
 - glucose_start sampled in GLUCOSE_START at every reset, so the agent also experiences being sated
 - actions rescaled to [-1, 1] (RescaleAction): PPO's Gaussian starts at mean 0, std 1, which maps to
   the middle of each command's range instead of half the samples clipping at move_cmd = 0
 - no VecNormalize: observations are already in [-1, 1] and the reward is scaled in the env
"""

import argparse
import functools
import json
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
from zfo.metabolism import GLUCOSE_RANGE, STARVATION_PENALTY
from zfo.policies import random_policy

DT = 0.1                          # s per step
HORIZON = 30.0                    # s; discounting time constant (~ the glucose half-life). This represents the agent's planning range. TODO can be informed by literature
GAMMA = math.exp(-DT / HORIZON)   # ~0.9967 at dt = 0.1; gamma is the actual discount factor, and we make it dt-independent by deriving it from a time constant. The agent sees the same planning horizon at any dt.
GLUCOSE_START = (20.0, 80.0)      # range for the starting glucose
N_ENVS = 8                        # parallel copies of the env
N_STEPS = 1024                    # steps per env per rollout (8192 per PPO update)
TOTAL_TIMESTEPS = 1_000_000       # ~28 simulated hours
SEED = 0
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"   # repo-root runs/, wherever this is run from
# default run configuration (2026-11-03): no observation flags, failed EATs cost a cooldown
# (trains reliably and its mean action also works; see docs/PORTING_NOTES.md)
HIDE_FLAGS = ("can_eat", "in_reach")
EAT_COST = "attempt"
EAT_COSTS = ("none", "attempt", "press")
FOOD_REGROWTH_INTERVAL = 10.0     # s per regrowth tick (0.1 pellets/s; staying at target needs ~0.13-0.15/s)
FOOD_PER_REGROWTH = 1             # pellets grown at each tick
SENSING_RANGE = None              # arena units; None: the arena diagonal (sees everything)
EPISODE_DURATION = 300.0          # s; long enough that the initial 20 pellets run out (~150 s) and regrowth matters
REWARD_MODE = "drive"             # "drive": distance from the target; "range": time inside GLUCOSE_RANGE
REWARD_MODES = ("drive", "range")


def run_env_kwargs(hide=HIDE_FLAGS, eat_cost=EAT_COST, food_regrowth_interval=FOOD_REGROWTH_INTERVAL,
                   episode_duration=EPISODE_DURATION, food_per_regrowth=FOOD_PER_REGROWTH,
                   sensing_range=SENSING_RANGE, reward_mode=REWARD_MODE, glucose_range=GLUCOSE_RANGE,
                   starvation_penalty=STARVATION_PENALTY):
    """
    The ZebrafishEnv kwargs of a run, with every observation flag, EAT rule, the food regrowth, the
    episode length and the reward set explicitly (saved as env_kwargs.json). hide: observation flags to
    leave out ("can_eat", "in_reach"); eat_cost: one of EAT_COSTS; food_regrowth_interval: s between
    regrowth ticks (None or 0: off); episode_duration: s; food_per_regrowth: pellets grown at each tick;
    sensing_range: arena units (None or 0: the arena diagonal, sees everything); reward_mode: one of
    REWARD_MODES; glucose_range, starvation_penalty: the range reward's (low, high) and one-off penalty
    (recorded only when reward_mode is "range").
    """
    if eat_cost not in EAT_COSTS:
        raise ValueError(f"eat_cost must be one of {EAT_COSTS}, got {eat_cost!r}")
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
    reward = {"reward_mode": reward_mode}
    if reward_mode == "range":
        reward.update({"glucose_range": list(glucose_range), "starvation_penalty": starvation_penalty})
    return {"observe_can_eat": "can_eat" not in hide, "observe_in_reach": "in_reach" not in hide,
            "eat_attempt_resets_cooldown": eat_cost == "attempt",
            "eat_press_resets_cooldown": eat_cost == "press",
            "food_regrowth_interval": food_regrowth_interval or None,
            "food_per_regrowth": food_per_regrowth,
            "sensing_range": sensing_range or None,
            "episode_duration": episode_duration, **reward}


def make_env(seed=None, **env_kwargs):
    """
    The env in physical action units, as the random baseline uses it.
    env_kwargs (e.g. observe_in_reach=False) are passed on to ZebrafishEnv.
    """
    return ZebrafishEnv(dt=DT, glucose_start=GLUCOSE_START, seed=seed, **env_kwargs)


def make_ppo_env(seed=None, **env_kwargs):
    """
    The env as PPO sees it, in training and evaluation alike: actions in [-1, 1], mapped by
    RescaleAction to the physical ranges. The only place that translation is defined.
    """
    return RescaleAction(make_env(seed, **env_kwargs), min_action=-1.0, max_action=1.0)


def load_env_kwargs(name):
    """
    The env_kwargs runs/NAME was trained with (env_kwargs.json); {} for runs saved before it existed,
    which used the defaults. Pass them to make_ppo_env so the observation layout matches the model.
    """
    path = os.path.join(RUNS_DIR, name, "env_kwargs.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


class HomeostasisCallback(BaseCallback):
    """
    Logs the summarized episode_metrics of the training episodes that ended in each rollout, under
    "homeostasis/" in SB3's logger (printed and written to progress.csv).
    Also keeps every training episode, one row each: "timestep" (total env steps, over all envs,
    when it ended), "env" (which parallel env played it) and its episode_metrics. Saved to
    save_path (.npz, {key: array (n_episodes,)}) after every rollout, so a crashed run keeps them.
    """

    # stores the physical parameters of the env that episode_metrics needs, and where to save episodes
    def __init__(self, dt, glucose_target, episode_duration, eat_cooldown, save_path, glucose_range=GLUCOSE_RANGE):
        super().__init__()
        self.dt = dt
        self.glucose_target = glucose_target
        self.glucose_range = glucose_range
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
                                          self.episode_duration, self.eat_cooldown, self.glucose_range)
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
    parser.add_argument("--hide", nargs="*", default=list(HIDE_FLAGS), choices=["can_eat", "in_reach"],
                        help="observation flags to leave out (default: both; --hide alone observes both)")
    parser.add_argument("--eat-cost", default=EAT_COST, choices=EAT_COSTS,
                        help="what a failed EAT costs (default: attempt)")
    parser.add_argument("--food-regrowth-interval", type=float, default=FOOD_REGROWTH_INTERVAL,
                        help="s between regrowth ticks (default: 10; 0 = no regrowth)")
    parser.add_argument("--food-per-regrowth", type=int, default=FOOD_PER_REGROWTH,
                        help="pellets grown at each regrowth tick (default: 1)")
    parser.add_argument("--episode-duration", type=float, default=EPISODE_DURATION,
                        help="episode time limit in s (default: 300)")
    parser.add_argument("--sensing-range", type=float, default=SENSING_RANGE,
                        help="pellets farther than this are not sensed (default: arena diagonal; 0 = same)")
    parser.add_argument("--reward", default=REWARD_MODE, choices=REWARD_MODES,
                        help="drive: distance from the target (default); range: time inside --glucose-range")
    parser.add_argument("--glucose-range", type=float, nargs=2, default=list(GLUCOSE_RANGE),
                        metavar=("LOW", "HIGH"), help="homeostatic range for --reward range (default: 40 60)")
    parser.add_argument("--starvation-penalty", type=float, default=STARVATION_PENALTY,
                        help="one-off penalty on starving for --reward range (default: 100)")
    args = parser.parse_args()
    run_dir = os.path.join(RUNS_DIR, args.name)
    env_kwargs = run_env_kwargs(args.hide, args.eat_cost, args.food_regrowth_interval, args.episode_duration,
                                args.food_per_regrowth, args.sensing_range, args.reward, args.glucose_range,
                                args.starvation_penalty)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "env_kwargs.json"), "w") as f:
        json.dump(env_kwargs, f)
    env_fn = functools.partial(make_env, **env_kwargs)
    ppo_env_fn = functools.partial(make_ppo_env, **env_kwargs)

    vec_env = make_vec_env(ppo_env_fn, n_envs=N_ENVS, seed=SEED) # creates N_ENVS copies of the env, each with a different seed
    model = PPO("MlpPolicy", vec_env, n_steps=N_STEPS, batch_size=256, gamma=GAMMA, gae_lambda=0.95,
                seed=SEED, device="cpu", verbose=0)   # small MLPs: CPU is as fast as MPS for PPO
    model.set_logger(configure(run_dir, ["stdout", "csv"]))
    probe = env_fn()
    # callback: runs at every env step and collects step info
    # keeps the metrics of every episode (saved to train_episodes.npz each rollout)
    # at the end of each rollout it logs the summarized episode_metrics of the episodes that finished
    callback = HomeostasisCallback(probe.dt, probe.glucose_target, probe.max_steps * probe.dt,
                                   probe.eat_cooldown, os.path.join(run_dir, "train_episodes.npz"),
                                   probe.glucose_range)
    model.learn(args.timesteps, callback=callback) # does the whole training loop, until args.timesteps env steps have been collected
    model.save(os.path.join(run_dir, "model"))

    evals = {
        "random": evaluate(random_policy(SEED), env_fn),
        "ppo": evaluate(ppo_policy(model), ppo_env_fn),
    }
    # per-episode metrics, keys "policy/metric": load with np.load(runs/NAME/eval.npz)
    np.savez(os.path.join(run_dir, "eval.npz"),
             **{f"{name}/{key}": values for name, per_episode in evals.items()
                for key, values in per_episode.items()})
    print_comparison({name: summarize(per_episode) for name, per_episode in evals.items()})
