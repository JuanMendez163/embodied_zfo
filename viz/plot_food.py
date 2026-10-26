"""
Visual sanity check for food placement in arena.py: spawn food for several seeds and plot it.
Run with: python viz/plot_food.py  (saves figures/food_check.png)

Each seed follows the episode-start order from arena.py: reset -> place agent -> spawn food
away from the agent. The last two panels pool many seeds to check placement is uniform:
 - absolute: pellets should fill the margin-shrunk arena evenly
 - relative to the agent: there should be an empty disc of radius food_min_distance around (0, 0)
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from viz.style import FIGURES_DIR
from zfo.arena import Arena

N_SEEDS = 6
N_POOLED = 500        # seeds pooled in the last two panels
AGENT_MARGIN = 5.0    # keep the agent's start this far from the walls

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
AGENT = "#2a78d6"
FOOD = "#1baf7a"


def episode_start(arena, seed):
    """
    Reset the arena, place the agent, then spawn food away from it. Returns the agent position.
    """
    arena.reset(seed)
    agent = arena.random_position(margin=AGENT_MARGIN)
    arena.spawn_food(avoid=agent)
    return agent


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=9, color=TEXT, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_constraints(ax, arena, agent):
    """
    Dashed outlines of where food may not go: the wall margin and the disc around the agent.
    """
    m = arena.food_wall_margin
    ax.add_patch(plt.Rectangle((m, m), arena.width - 2 * m, arena.height - 2 * m, fill=False,
                               edgecolor=TEXT_MUTED, linewidth=0.8, linestyle=":"))
    ax.add_patch(plt.Circle(agent, arena.food_min_distance, fill=False,
                            edgecolor=AGENT, linewidth=1, linestyle="--"))


def main():
    arena = Arena(size=(100.0, 100.0))
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.6), facecolor=SURFACE)
    axes = axes.flat

    for seed in range(N_SEEDS):
        ax = next(axes)
        agent = episode_start(arena, seed)
        arena.draw(ax, edgecolor=TEXT_MUTED, food_color=FOOD)
        draw_constraints(ax, arena, agent)
        ax.scatter(*agent, s=60, color=AGENT, zorder=3)
        distances = np.linalg.norm(arena.food_positions - agent, axis=1)
        style(ax, f"seed {seed}: {arena.num_food} pellets, nearest {distances.min():.1f} away")

    # pooled: absolute positions
    absolute, relative = [], []
    for seed in range(N_POOLED):
        agent = episode_start(arena, seed)
        absolute.append(arena.food_positions)
        relative.append(arena.food_positions - agent)
    absolute, relative = np.concatenate(absolute), np.concatenate(relative)

    ax = next(axes)
    arena.reset()  # empty arena, so draw() shows only the walls
    arena.draw(ax, edgecolor=TEXT_MUTED)
    ax.scatter(*absolute.T, s=2, color=FOOD, alpha=0.35, edgecolor="none")
    style(ax, f"{N_POOLED} seeds pooled ({len(absolute)} pellets):\nshould be even inside the margin")

    ax = next(axes)
    ax.scatter(*relative.T, s=2, color=FOOD, alpha=0.35, edgecolor="none")
    ax.add_patch(plt.Circle((0, 0), arena.food_min_distance, fill=False,
                            edgecolor=AGENT, linewidth=1, linestyle="--"))
    ax.scatter(0, 0, s=60, color=AGENT, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    style(ax, f"Same pellets relative to the agent (zoomed):\nempty disc of radius {arena.food_min_distance:g}")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=AGENT, markersize=8, label="agent start"),
        plt.Line2D([], [], marker="o", linestyle="none", color=FOOD, markersize=5, label="food pellet"),
        plt.Line2D([], [], color=AGENT, linestyle="--", label=f"no food within {arena.food_min_distance:g} of agent"),
        plt.Line2D([], [], color=TEXT_MUTED, linestyle=":", label=f"wall margin ({arena.food_wall_margin:g})"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=9, labelcolor=TEXT_MUTED)
    fig.suptitle("arena.py: food placement across seeds", fontsize=12, color=TEXT, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "food_check.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
