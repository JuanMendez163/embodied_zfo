"""
Visual sanity checks for movement.py: roll out a few command sequences and plot the trajectories.
Run with: python plot_movement.py  (saves figures/movement_check.png)
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np

from movement import step_pose

ARENA = (100.0, 100.0)
BODY_RADIUS = 1.0
MAX_SPEED = 10.0     # units/s
MAX_TURN = math.pi   # rad/s
DT = 0.1             # s

# colors (reference palette, light mode)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
PATH = "#2a78d6"     # trajectory
WALL_HIT = "#eb6834" # steps where enforce_bounds clipped the position


def rollout(position, heading, commands):
    """
    Apply a sequence of (move_cmd, turn_cmd) and record every pose.
    Returns (positions [n+1, 2], headings [n+1], hit_wall [n]).
    """
    positions, headings, hits = [np.array(position, dtype=float)], [heading], []
    for move_cmd, turn_cmd in commands:
        pos, heading, hit_wall = step_pose(positions[-1], headings[-1], move_cmd, turn_cmd,
                                           MAX_SPEED, MAX_TURN, DT, ARENA, BODY_RADIUS)
        positions.append(pos)
        headings.append(heading)
        hits.append(hit_wall)
    return np.array(positions), np.array(headings), np.array(hits)


def smooth_random_commands(rng, n_steps, smoothing=0.9):
    """
    Commands that drift instead of jumping: each step blends the previous command with fresh noise.
    move_cmd is squashed to [0, 1] and turn_cmd to [-1, 1], as wef's actor does.
    """
    raw = np.zeros(2)
    commands = []
    for _ in range(n_steps):
        raw = smoothing * raw + (1 - smoothing) * rng.normal(0, 3, size=2)
        commands.append((1 / (1 + math.exp(-raw[0])), math.tanh(raw[1])))
    return commands


def scenarios():
    circle_turn = 0.2
    n_circle = round(2 * math.pi / (circle_turn * MAX_TURN * DT))
    radius = MAX_SPEED / (circle_turn * MAX_TURN)
    return [
        ("Straight line (move=1, turn=0)",
         f"Heading 30°, 60 steps × {MAX_SPEED * DT:.0f} unit = 60 units in a straight line",
         rollout([20, 20], math.radians(30), [(1.0, 0.0)] * 60)),
        (f"Circle (move=1, turn={circle_turn})",
         f"{n_circle} steps of {math.degrees(circle_turn * MAX_TURN * DT):.1f}°, radius ≈ {radius:.0f}; should close on the start",
         rollout([50, 30], 0.0, [(1.0, circle_turn)] * n_circle)),
        ("Wall slide (move=1, turn=0)",
         "Heading 60° into the right wall; x stops, y keeps moving",
         rollout([90, 20], math.radians(60), [(1.0, 0.0)] * 40)),
        ("Smooth random commands",
         "300 steps, seed 0; should look like a swimming path",
         rollout([50, 50], 0.0, smooth_random_commands(np.random.default_rng(0), 300))),
    ]


def plot_panel(ax, title, subtitle, result):
    positions, headings, hits = result
    width, height = ARENA

    ax.add_patch(plt.Rectangle((0, 0), width, height, fill=False, edgecolor=TEXT_MUTED, linewidth=1))
    ax.plot(positions[:, 0], positions[:, 1], color=PATH, linewidth=2, solid_capstyle="round", zorder=2)

    # heading arrows at a few evenly spaced poses
    idx = np.linspace(0, len(positions) - 1, 8).astype(int)
    ax.quiver(positions[idx, 0], positions[idx, 1], np.cos(headings[idx]), np.sin(headings[idx]),
              color=PATH, angles="xy", scale_units="xy", scale=0.33, width=0.005, zorder=3)

    # hits[i] belongs to the pose after step i, i.e. positions[i + 1]
    hit_positions = positions[1:][hits]
    if len(hit_positions):
        ax.scatter(hit_positions[:, 0], hit_positions[:, 1], s=36, color=WALL_HIT,
                   edgecolor=SURFACE, linewidth=1.5, zorder=4, label="hit_wall=True")
        ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=TEXT_MUTED)

    ax.scatter(*positions[0], s=64, color=SURFACE, edgecolor=TEXT, linewidth=1.5, zorder=5)
    ax.annotate("start", positions[0], xytext=(6, -12), textcoords="offset points", fontsize=8, color=TEXT_MUTED)

    ax.set_title(f"{title}\n", fontsize=10, color=TEXT, loc="left")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8, color=TEXT_MUTED)
    ax.set_xlim(-3, width + 3)
    ax.set_ylim(-3, height + 3)
    ax.set_aspect("equal")
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10, 10.5), facecolor=SURFACE)
    for ax, (title, subtitle, result) in zip(axes.flat, scenarios()):
        ax.set_facecolor(SURFACE)
        plot_panel(ax, title, subtitle, result)
    fig.suptitle("movement.py: trajectories under fixed and random commands", fontsize=12,
                 color=TEXT, x=0.06, ha="left")
    fig.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out = os.path.join("figures", "movement_check.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
