"""
Learning curves of several training runs on shared axes (runs/NAME/train_episodes.npz each), e.g. the
observation-flag ablation.
Run with: python viz/compare_training.py --names RUN ... [--labels LABEL ...] [--zoom N] [--out NAME]
(saves figures/compare_OUT.png and figures/compare_OUT_homeostasis.png)

compare_OUT.png: survival time, fraction of episodes starved, pellets eaten. Top row: the whole run;
bottom row: the first --zoom env steps, with the rollout boundaries (one PPO update each), to count how
many updates a run needed.
compare_OUT_homeostasis.png: glucose target error and EAT attempt rate, whole run only. (EAT efficiency
is left out: on training episodes, sampled EAT flickers into short bouts, so it is noise-dominated; see
docs/PORTING_NOTES.md.)
Episodes are grouped by rollout as in viz/plot_training.py; the per-episode dots are left out so the
runs stay readable.
"""

import argparse
import os

import matplotlib.pyplot as plt

from training.train_ppo import N_ENVS
from viz.plot_training import STEPS_PER_ROLLOUT, load_train_episodes, plot_learning_curve
from viz.style import FIGURES_DIR, RUN_COLORS, TEXT_MUTED

# (episode_metrics key, y label, y scale)
SURVIVAL_METRICS = [("time", "survival time (s)", "linear"),
                    ("starved", "fraction of episodes starved", "linear"),
                    ("total_eaten", "pellets eaten", "linear")]
HOMEOSTASIS_METRICS = [("glucose_target_error", "glucose target error (log)", "log"),
                       ("eat_attempt_rate", "EAT attempt rate (fraction of time held)", "linear")]


def plot_comparison(runs, metrics, zoom=None):
    """
    Figure comparing runs = {label: train episodes (load_train_episodes)}, one column per metrics entry:
    one row for the whole run, plus (zoom = env steps) one for the first zoom steps with the rollout
    boundaries. Colors follow RUN_COLORS in order.
    """
    if len(runs) > len(RUN_COLORS):
        raise ValueError(f"at most {len(RUN_COLORS)} runs per figure, got {len(runs)}")
    rows = [None] if zoom is None else [None, zoom]
    fig, axes = plt.subplots(len(rows), len(metrics), figsize=(5 * len(metrics), 3.75 * len(rows)),
                             squeeze=False)
    for row, xmax in enumerate(rows):
        for col, (key, ylabel, yscale) in enumerate(metrics):
            for color, (label, episodes) in zip(RUN_COLORS, runs.items()):
                plot_learning_curve(axes[row, col], episodes, key, ylabel, xmax=xmax, color=color,
                                    label=label, dots=False)
            axes[row, col].set_yscale(yscale)
    axes[0, 0].legend(frameon=False, fontsize=9)
    if zoom is not None:
        axes[1, 0].set_title(f"first {zoom:,} steps; vertical lines = rollout boundaries "
                             f"({STEPS_PER_ROLLOUT:,} steps each)", fontsize=9, loc="left", color=TEXT_MUTED)
    fig.suptitle(f"{N_ENVS} envs per run; line = mean per rollout, band = ±1 standard error")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", required=True, help="runs/NAME directories to compare")
    parser.add_argument("--labels", nargs="+", help="legend labels, one per name (default: the names)")
    parser.add_argument("--zoom", type=int, default=100_000, help="env steps shown in the zoomed row")
    parser.add_argument("--out", default="runs", help="figure names: figures/compare_OUT*.png")
    args = parser.parse_args()
    labels = args.labels or args.names
    if len(labels) != len(args.names):
        parser.error("--labels needs one label per --names entry")

    runs = {label: load_train_episodes(name) for label, name in zip(labels, args.names)}
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for suffix, fig in [("", plot_comparison(runs, SURVIVAL_METRICS, args.zoom)),
                        ("_homeostasis", plot_comparison(runs, HOMEOSTASIS_METRICS))]:
        path = FIGURES_DIR / f"compare_{args.out}{suffix}.png"
        fig.savefig(path, dpi=150)
        print(f"saved {path}")
