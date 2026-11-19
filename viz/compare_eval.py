"""
Final-model comparison: the evaluation episodes (runs/NAME/eval.npz) of several trained runs side by
side, one column per run (or per group of runs). Every run is evaluated on the same seeded episodes
(training/metrics.py, EVAL_SEED), so the columns differ only in the model and its env settings.
Run with: python viz/compare_eval.py --names RUN ... [--labels LABEL ...] [--out OUT]
(saves figures/compare_OUT_eval.png, next to compare_training.py's compare_OUT.png)

Runs given the same label are pooled as seeds of one setting: the dots are all their episodes and the
black marker is the mean over the pooled episodes (± standard error over episodes, which leaves out
seed-to-seed variation).
A "\\n" in a label starts a new line. The dotted line in each column is the random policy's mean on
the same episodes and env settings (its seed is fixed, so it is the same for every seed of a setting).
Panels: survival time, fraction starved (no dots: each episode is 0 or 1), pellets eaten, time in the
glucose range and mean glucose (time after starvation counts as 0; shaded: the default glucose range).
A run whose eval.npz lacks a metric (saved before it existed) is left out of that panel.
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from training.train_ppo import RUNS_DIR
from viz.style import AGENT, FIGURES_DIR, GRID, SURFACE, TEXT, TEXT_MUTED
from zfo.metabolism import GLUCOSE_RANGE

# (episode_metrics key, y label)
EVAL_METRICS = [("time", "survival time (s)"),
                ("starved", "fraction of episodes starved"),
                ("total_eaten", "pellets eaten"),
                ("in_range_time", "time in glucose range (s)"),
                ("glucose_mean", "mean glucose")]
NO_DOTS = ("starved",)   # 0/1 per episode: dots would only pile up at 0 and 1


def load_eval(name):
    """
    The evaluation episodes of runs/NAME: {policy: {metric: (n_episodes,) array}}, policies "ppo" and
    "random" (eval.npz keys are "policy/metric").
    """
    data = np.load(os.path.join(RUNS_DIR, name, "eval.npz"))
    evals = {}
    for key in data.files:
        policy, metric = key.split("/", 1)
        evals.setdefault(policy, {})[metric] = data[key]
    return evals


def group_runs(names, labels):
    """
    {label: [load_eval(name), ...]} in order of first appearance; names sharing a label (seeds of one
    setting) are grouped under it.
    """
    groups = {}
    for name, label in zip(names, labels):
        groups.setdefault(label, []).append(load_eval(name))
    return groups


def mean_se(values):
    """
    Mean and standard error of values, ignoring nan (episodes where a ratio metric is undefined).
    """
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    se = values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return values.mean(), se


def plot_group(ax, x, evals, key, rng):
    """
    Draws one column of a panel at x for the seeds evals of one setting and metric key: per-episode dots
    (jittered with rng), pooled mean ± standard error and the random policy's mean (dotted). Draws
    nothing if a run lacks key.
    """
    if any(key not in e["ppo"] for e in evals):   # run saved before this metric existed
        return
    ppo = np.concatenate([e["ppo"][key] for e in evals])
    if key not in NO_DOTS:
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(ppo)), ppo, s=9, color=AGENT, alpha=0.35,
                   linewidths=0, zorder=2)
    mean, se = mean_se(ppo)
    ax.errorbar(x, mean, yerr=se, fmt="o", color=TEXT, markersize=5, capsize=3, zorder=4)
    random_mean = mean_se(np.concatenate([e["random"][key] for e in evals]))[0]
    ax.hlines(random_mean, x - 0.35, x + 0.35, color=TEXT_MUTED, linestyles=":", linewidth=1.3, zorder=1)


def plot_eval_comparison(groups, metrics=EVAL_METRICS, ncols=3):
    """
    Figure comparing groups = {label: [eval of each seed]} (group_runs), one panel per metrics entry and
    one column per group: per-episode dots, pooled mean ± standard error and the random policy's mean
    (dotted).
    """
    nrows = math.ceil(len(metrics) / ncols)
    panel_width = max(4.0, 1.1 * len(groups) + 1.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel_width, nrows * 3.8), squeeze=False)
    rng = np.random.default_rng(0)   # jitter only; fixed so the figure is reproducible
    for ax, (key, ylabel) in zip(axes.flat, metrics):
        for x, evals in enumerate(groups.values()):
            plot_group(ax, x, evals, key, rng)
        if key == "glucose_mean":
            ax.axhspan(*GLUCOSE_RANGE, color=GRID, alpha=0.6, zorder=0)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(list(groups), fontsize=8)   # long labels: break them with "\n"
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    handles = [Line2D([], [], marker="o", linestyle="none", color=AGENT, alpha=0.35, markersize=4,
                      label="evaluation episode"),
               Line2D([], [], marker="o", linestyle="none", color=TEXT, markersize=5,
                      label="mean ± SE over episodes"),
               Line2D([], [], linestyle=":", color=TEXT_MUTED, label="random policy mean")]
    n_episodes = len(next(iter(groups.values()))[0]["ppo"]["time"])
    fig.suptitle(f"Final models on the same {n_episodes} evaluation episodes (deterministic actions)")
    if len(metrics) < axes.size:   # legend in the first empty slot of the grid
        axes.flat[len(metrics)].legend(handles=handles, loc="center", frameon=False, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False,
                   fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", required=True, help="runs/NAME directories to compare")
    parser.add_argument("--labels", nargs="+",
                        help="column labels, one per name (default: the names); repeated labels pool seeds")
    parser.add_argument("--out", default="runs", help="figure name: figures/compare_OUT_eval.png")
    args = parser.parse_args()
    labels = [label.replace("\\n", "\n") for label in (args.labels or args.names)]
    if len(labels) != len(args.names):
        parser.error("--labels needs one label per --names entry")

    fig = plot_eval_comparison(group_runs(args.names, labels))
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = FIGURES_DIR / f"compare_{args.out}_eval.png"
    fig.savefig(path, dpi=150)
    print(f"saved {path}")
