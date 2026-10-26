"""
Train a PPO actor-critic (Stable-Baselines3) to keep glucose near the target, then compare it with a
random policy and the scripted forager (animate_env.forage_action) on the same evaluation episodes.
Run with: python train_ppo.py [--timesteps N] [--name NAME]  (writes runs/NAME/: model, progress.csv)

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

import numpy as np
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from animate_env import forage_action
from env import ZebrafishEnv

DT = 0.1                          # s per step
HORIZON = 30.0                    # s; discounting time constant (~ the glucose half-life). This represents the agent's planning range. TODO can be informed by literature
GAMMA = math.exp(-DT / HORIZON)   # ~0.9967 at dt = 0.1; gamma is the actual discount factor, and we make it dt-independent by deriving it from a time constant. The agent sees the same planning horizon at any dt.
GLUCOSE_START = (20.0, 80.0)      # range for the starting glucose
N_ENVS = 8                        # parallel copies of the env
N_STEPS = 1024                    # steps per env per rollout (8192 per PPO update)
TOTAL_TIMESTEPS = 1_000_000       # ~28 simulated hours
SEED = 0
EVAL_EPISODES = 30
EVAL_SEED = 10_000                # evaluation episodes use seeds the training envs never see


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


def episode_metrics(infos, dt, glucose_target, episode_duration, eat_cooldown):
    """
    Summary of how well one episode kept homeostasis, as a dict of floats (all averaged over episodes).
    infos: the env's info dict from every step of the episode, in order. Each has "glucose", "action"
    (move_cmd, turn_cmd, eat_cmd as applied), "eat_attempted", "n_eaten", and "eat_ready" (cooldown
    passed) / "food_in_reach" (pellet in the mouth cone), both at the start of the step, i.e. what the
    agent saw as can_eat / in_reach when it chose the action; the last one also has
    "total_eaten", "end_reason" and "time" (s) for the whole episode. dt in s.
    episode_duration: the time limit (s), so time after starvation can be counted (as glucose 0) and
    every episode is averaged over the same duration.
    eat_cooldown: s between successful eats, which caps the eating rate at 1 / eat_cooldown.
    Note: metrics with a raw step count are NOT dt-independent, be wary
    """
    glucose_trace = np.array([i["glucose"] for i in infos])
    actions = np.array([i["action"] for i in infos])     # (steps, 3)
    attempts = np.array([i["eat_attempted"] for i in infos])
    ready = np.array([i["eat_ready"] for i in infos])
    reach = np.array([i["food_in_reach"] for i in infos])
    info = infos[-1]
    n_attempts = attempts.sum() # the total number of steps where the agent attempted to eat

    # EAT bouts (runs of consecutive steps with EAT held): a bout lasting D s allows at most
    # ceil(D / eat_cooldown) eats (one at its start, then one per cooldown), so possible_eats bounds
    # total_eaten. 1e-9 keeps a bout of exactly one cooldown from rounding up to 2.
    starts = np.flatnonzero(attempts & ~np.r_[False, attempts[:-1]])     # first step of each bout
    ends = np.flatnonzero(attempts & ~np.r_[attempts[1:], False])        # last step of each bout
    bout_durations = (ends - starts + 1) * dt                    # s
    possible_eats = np.sum(np.ceil(bout_durations / eat_cooldown - 1e-9))

    # time after starvation counts at error = target, and every episode is averaged over episode_duration
    # units are error * s
    error_alive = np.sum(np.abs(glucose_trace - glucose_target)) * dt
    error_dead = max(0.0, episode_duration - info["time"]) * glucose_target

    return {
        "starved": float(info["end_reason"] == "starved"),
        # mean and std are over alive time only, so misleading for starved episodes; read with "starved"
        "glucose_mean": float(np.mean(glucose_trace)),
        "glucose_std": float(np.std(glucose_trace)),
        "glucose_target_error": float((error_alive + error_dead) / episode_duration),
        "total_eaten": float(info["total_eaten"]),
        "time": float(info["time"]),
        "mean_move_cmd": float(np.mean(actions[:, 0])),
        "mean_abs_turn_cmd": float(np.mean(np.abs(actions[:, 1]))),
        "eat_attempt_rate": float(n_attempts / len(infos)),                # fraction of time EAT is held
        "eat_efficiency": float(info["total_eaten"] / possible_eats) if possible_eats else 0.0,
        # why EAT presses fail, as fractions of EAT-held time (reach first: no food at the mouth,
        # then food at the mouth but still cooling down); the rest (~eats * dt) are successful presses
        "eat_out_of_reach": float(np.sum(attempts & ~reach) / n_attempts) if n_attempts else 0.0,
        "eat_during_cooldown": float(np.sum(attempts & reach & ~ready) / n_attempts) if n_attempts else 0.0,
    }


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


def evaluate(policy, env_fn=make_env, n_episodes=EVAL_EPISODES, seed=EVAL_SEED):
    """
    Mean episode_metrics of policy(env, obs) -> action over n_episodes of env_fn() (make_env, or
    make_ppo_env for PPO, whose actions are in [-1, 1]).
    Episode i is reset with seed + i, so every policy faces the same arenas and starting glucose.
    """
    env = env_fn()
    base = env.unwrapped
    results = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        trace, done = [], False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(env, obs))
            trace.append(info)
            done = terminated or truncated
        results.append(episode_metrics(trace, base.dt, base.glucose_target, base.max_steps * base.dt,
                                       base.eat_cooldown))
    return {key: float(np.mean([r[key] for r in results])) for key in results[0]}


def ppo_policy(model):
    """
    Deterministic (mean) PPO action, in [-1, 1]; evaluate it on make_ppo_env.
    """
    return lambda env, obs: model.predict(obs, deterministic=True)[0]


def random_policy(seed=SEED):
    """
    Uniformly random commands, a lower bound.
    """
    rng = np.random.default_rng(seed)
    return lambda env, obs: rng.uniform(env.action_space.low, env.action_space.high)


def print_comparison(rows):
    """
    Table of {policy name: metrics dict}, one row per policy.
    """
    keys = list(next(iter(rows.values())))
    widths = [max(len(k), 8) + 2 for k in keys]
    print(f"\n{'policy':<10}" + "".join(f"{k:>{w}}" for k, w in zip(keys, widths)))
    for name, metrics in rows.items():
        print(f"{name:<10}" + "".join(f"{metrics[k]:>{w}.3f}" for k, w in zip(keys, widths)))


if __name__ == "__main__":

    # Reads command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--name", default="ppo")
    args = parser.parse_args()
    run_dir = os.path.join("runs", args.name)

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
        "random": evaluate(random_policy()),
        "scripted": evaluate(lambda env, obs: forage_action(env)),
        "ppo": evaluate(ppo_policy(model), make_ppo_env),
    })
