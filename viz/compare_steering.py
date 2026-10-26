"""
Compare WaypointFish speed modes on the same rectangle, with the same noise seed.
Run with: python viz/compare_steering.py  (saves figures/steering_compare.png and .mp4)

Each variant changes one thing relative to the default ("coupled", arrive_radius=4):
 - speed_mode="cruise": keeps >= 30% speed through turns instead of pivoting
 - speed_mode="cosine": speed set by cos(heading_error) rather than by turn_cmd
 - arrive_radius=2:     switches waypoints later, so corners are cut less
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from viz.animate_movement import (AGENT, DT, FPS, N_STEPS, RECTANGLE_CORNERS, SURFACE, TEXT, TEXT_MUTED,
                                  AgentPanel, WaypointFish, rollout)
from viz.style import FIGURES_DIR
from zfo.arena import Arena

SEED = 1
VARIANTS = [
    ("coupled (yours)", {}),
    ("cruise ≥ 0.3", {"speed_mode": "cruise"}),
    ("cosine", {"speed_mode": "cosine"}),
    ("coupled, arrive_radius=2", {"arrive_radius": 2.0}),
]


def run_variants(arena):
    runs = []
    for label, kwargs in VARIANTS:
        fish = WaypointFish(RECTANGLE_CORNERS, np.random.default_rng(SEED), **kwargs)
        run = rollout(arena, [30, 35], 0.0, fish)
        run["waypoints"] = np.array(RECTANGLE_CORNERS)
        run["corners_reached"] = fish.corners_reached
        runs.append((label, run))
    return runs


def summarize(run):
    """
    Distance swum and how many near-stops (move_cmd < 0.1).
    """
    positions, commands = run["positions"], run["commands"][1:]
    distance = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
    stops = int((commands[:, 0] < 0.1).sum())
    return distance, stops


def plot_static(runs):
    fig, axes = plt.subplots(2, len(runs), figsize=(4 * len(runs), 7), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    rectangle = np.array(RECTANGLE_CORNERS + [RECTANGLE_CORNERS[0]])
    for col, (label, run) in enumerate(runs):
        distance, stops = summarize(run)
        positions = run["positions"]

        ax = axes[0, col]
        ax.set_facecolor(SURFACE)
        ax.plot(*rectangle.T, color=TEXT_MUTED, linewidth=0.8, linestyle="--")
        ax.plot(positions[:, 0], positions[:, 1], color=AGENT, linewidth=1.5)
        ax.scatter(*positions[0], s=40, color=SURFACE, edgecolor=TEXT, zorder=3)
        ax.set_xlim(22, 78)
        ax.set_ylim(27, 63)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{label}\ndistance {distance:.0f}, corners {run['corners_reached']}, near-stops {stops}",
                     fontsize=9, color=TEXT, loc="left")

        ax = axes[1, col]
        ax.set_facecolor(SURFACE)
        t = np.arange(1, len(run["commands"])) * DT
        ax.plot(t, run["commands"][1:, 0], color=AGENT, linewidth=1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("time (s)", fontsize=8, color=TEXT_MUTED)
        if col == 0:
            ax.set_ylabel("move_cmd", fontsize=8, color=TEXT_MUTED)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(f"WaypointFish steering variants, {N_STEPS} steps, same noise (seed {SEED}); dashed = target rectangle",
                 fontsize=11, color=TEXT, x=0.02, ha="left")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "steering_compare.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"saved {out}")


def animate(arena, runs):
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 5.2), facecolor=SURFACE)
    panels = [AgentPanel(ax, arena, label, run) for ax, (label, run) in zip(axes, runs)]
    fig.suptitle("WaypointFish steering variants (same noise)", fontsize=11, color=TEXT, x=0.02, ha="left")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.17, wspace=0.08)

    def update(t):
        return [artist for panel in panels for artist in panel.update(t)]

    anim = FuncAnimation(fig, update, frames=N_STEPS + 1, blit=True)
    out = os.path.join(FIGURES_DIR, "steering_compare.mp4")
    anim.save(out, writer=FFMpegWriter(fps=FPS), dpi=120, savefig_kwargs={"facecolor": SURFACE})
    print(f"saved {out}")


def main():
    arena = Arena(size=(100.0, 100.0))
    os.makedirs(FIGURES_DIR, exist_ok=True)
    runs = run_variants(arena)
    plot_static(runs)
    animate(arena, runs)


if __name__ == "__main__":
    main()
