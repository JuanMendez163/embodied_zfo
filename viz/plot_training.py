"""
Learning curves from a training run's per-episode metrics (runs/NAME/train_episodes.npz, written by
training/train_ppo.py's HomeostasisCallback).
Run with: python viz/plot_training.py [--name NAME]  (saves figures/training_NAME.png)

The x-axis is training experience: total env steps over all parallel envs when an episode ended. With
N_ENVS envs the episodes do not form one sequence, but every rollout (N_STEPS per env) is collected with
one fixed policy, so episodes ending in the same rollout are samples of the same policy version and are
grouped by rollout. An episode (episode_duration / dt steps) can span a policy update, so the curve is
blurred over about one update.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from training.train_ppo import N_ENVS, N_STEPS, RUNS_DIR
from viz.style import AGENT, FIGURES_DIR, GRID, SURFACE, TEXT_MUTED

STEPS_PER_ROLLOUT = N_ENVS * N_STEPS   # env steps per PPO update, over all envs


def load_train_episodes(name):
    """
    {metric: array (n_episodes,)} for every training episode of runs/NAME, including "timestep" (total
    env steps when it ended) and "env" (which parallel env played it).
    """
    with np.load(os.path.join(RUNS_DIR, name, "train_episodes.npz")) as data:
        return {key: data[key] for key in data.files}


def rollout_index(timestep, steps_per_rollout=STEPS_PER_ROLLOUT):
    """
    Which rollout (policy version) each episode ended in. timestep counts the step that ended the episode,
    so rollout k covers timesteps k * steps_per_rollout + 1 ... (k + 1) * steps_per_rollout.
    """
    return (np.asarray(timestep) - 1) // steps_per_rollout


def rollout_summary(rollout, values):
    """
    One summary point per rollout that ended at least one episode: (rollouts (n,), center (n,),
    low (n,), high (n,)), where center is the line drawn and low..high the shaded band.
    """
    rollouts, center, low, high = [], [], [], []
    for k in np.unique(rollout):
        mask = rollout == k
        rollouts.append(k)
        center.append(np.mean(values[mask]))
        standard_error = np.std(values[mask]) / np.sqrt(np.sum(mask)) # ddof=0
        low.append(np.mean(values[mask]) - standard_error)
        high.append(np.mean(values[mask]) + standard_error)
    return np.asarray(rollouts), np.asarray(center), np.asarray(low), np.asarray(high)


def plot_learning_curve(ax, episodes, key, ylabel, steps_per_rollout=STEPS_PER_ROLLOUT, xmax=None):
    """
    One metric over training: each episode as a faint dot at the timestep it ended, and its per-rollout
    summary (rollout_summary) as a line with a band, one marker per rollout, placed at the end of each
    rollout. With xmax, zooms to the first xmax env steps and draws the rollout boundaries.
    """
    timestep, values = episodes["timestep"], episodes[key]
    rollouts, center, low, high = rollout_summary(rollout_index(timestep, steps_per_rollout), values)
    x = (rollouts + 1) * steps_per_rollout   # end of the rollout: the experience the policy had
    if xmax is not None:
        ax.set_xlim(0, xmax)
        for boundary in np.arange(steps_per_rollout, xmax, steps_per_rollout):   # one policy update each
            ax.axvline(boundary, color=GRID, linewidth=0.8, zorder=0)
    ax.scatter(timestep, values, s=6 if xmax else 4, color=TEXT_MUTED, alpha=0.4 if xmax else 0.25,
               linewidths=0)
    ax.fill_between(x, low, high, color=AGENT, alpha=0.2, linewidth=0)
    ax.plot(x, center, color=AGENT, linewidth=1.5, marker="o", markersize=4 if xmax else 2)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y" if xmax else "both", color=GRID, linewidth=0.8)   # zoomed: verticals are rollouts
    ax.set_xlabel("env steps (all envs)")
    ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ppo")
    parser.add_argument("--zoom", type=int, default=100_000, help="env steps shown in the bottom row")
    args = parser.parse_args()

    episodes = load_train_episodes(args.name)
    # top row: the whole run; bottom row: the first --zoom env steps, with rollout boundaries
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, xmax in enumerate([None, args.zoom]):
        plot_learning_curve(axes[row, 0], episodes, "time", "survival time (s)", xmax=xmax)
        plot_learning_curve(axes[row, 1], episodes, "starved", "fraction of episodes starved", xmax=xmax)
    axes[1, 0].set_title(f"first {args.zoom:,} steps; vertical lines = rollout boundaries "
                         f"({STEPS_PER_ROLLOUT:,} steps each)", fontsize=9, loc="left", color=TEXT_MUTED)
    fig.suptitle(f"run {args.name}: {len(episodes['time'])} training episodes, {N_ENVS} envs; "
                 f"line = mean per rollout, band = ±1 standard error")
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = FIGURES_DIR / f"training_{args.name}.png"
    fig.savefig(path, dpi=150)
    print(f"saved {path}")
