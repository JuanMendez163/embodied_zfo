"""
Video sanity check for movement.py: animate an agent moving inside an Arena.
Run with: python animate_movement.py  (saves figures/movement.mp4)

Three scenarios play side by side, one step per frame:
 - circle:     constant move + turn, should loop forever on the same circle
 - wall slide: straight into a wall at an angle, should slide along it and stop in the corner
 - swimming:   smoothly varying random commands
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection

from arena import Arena
from movement import step_pose
from plot_movement import smooth_random_commands

MAX_SPEED = 10.0     # units/s
MAX_TURN = math.pi   # rad/s
DT = 0.1             # s
BODY_RADIUS = 2.0
N_STEPS = 300
FPS = 30             # 3x real time, since each step is DT = 0.1 s
TRAIL = 60           # number of past steps shown behind the agent

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
AGENT = "#2a78d6"
WALL_HIT = "#eb6834"


def rollout(arena, position, heading, commands):
    """
    Apply (move_cmd, turn_cmd) pairs inside the arena and record every step.
    Returns a dict of arrays, one entry per step (index 0 is the start pose).
    """
    positions, headings, hits = [np.array(position, dtype=float)], [heading], [False]
    for move_cmd, turn_cmd in commands:
        pos, heading, hit_wall = step_pose(positions[-1], headings[-1], move_cmd, turn_cmd,
                                           MAX_SPEED, MAX_TURN, DT, arena.size, BODY_RADIUS)
        positions.append(pos)
        headings.append(heading)
        hits.append(hit_wall)
    commands = [(0.0, 0.0)] + list(commands)  # no command produced the start pose
    return {
        "positions": np.array(positions),
        "headings": np.array(headings),
        "hits": np.array(hits),
        "commands": np.array(commands),
    }


def scenarios(arena):
    return [
        ("Circle (move=1, turn=0.2)",
         rollout(arena, [50, 30], 0.0, [(1.0, 0.2)] * N_STEPS)),
        ("Wall slide (move=1, turn=0, heading 60°)",
         rollout(arena, [70, 15], math.radians(60), [(1.0, 0.0)] * N_STEPS)),
        ("Swimming (smooth random commands)",
         rollout(arena, arena.random_position(margin=20), 0.0,
                 smooth_random_commands(arena.np_random, N_STEPS))),
    ]


class AgentPanel:
    """
    The artists for one scenario: arena walls, fading trail, agent body + heading, readout.
    """

    def __init__(self, ax, arena, title, run):
        self.run = run
        ax.set_facecolor(SURFACE)
        arena.draw(ax, edgecolor=TEXT_MUTED)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        self.trail = LineCollection([], linewidths=2, capstyle="round")
        ax.add_collection(self.trail)
        self.body = plt.Circle((0, 0), BODY_RADIUS, color=AGENT, zorder=3)
        ax.add_patch(self.body)
        (self.heading,) = ax.plot([], [], color=TEXT, linewidth=1.5, zorder=4)
        self.readout = ax.text(0.02, -0.02, "", transform=ax.transAxes, va="top",
                               fontsize=8, color=TEXT_MUTED, family="monospace")

    def update(self, t):
        positions, headings = self.run["positions"], self.run["headings"]
        hit_wall = self.run["hits"][t]
        move_cmd, turn_cmd = self.run["commands"][t]

        # trail: segments between the last TRAIL positions, fading out toward the oldest
        start = max(0, t - TRAIL)
        points = positions[start:t + 1]
        segments = np.stack([points[:-1], points[1:]], axis=1) if len(points) > 1 else []
        self.trail.set_segments(segments)
        n = len(segments)
        colors = np.tile(plt.matplotlib.colors.to_rgba(AGENT), (n, 1))
        colors[:, 3] = np.linspace(0.05, 0.8, n) if n else []
        self.trail.set_color(colors)

        x, y = positions[t]
        self.body.center = (x, y)
        self.body.set_color(WALL_HIT if hit_wall else AGENT)
        tip = positions[t] + 2.5 * BODY_RADIUS * np.array([math.cos(headings[t]), math.sin(headings[t])])
        self.heading.set_data([x, tip[0]], [y, tip[1]])

        self.readout.set_text(
            f"step {t:3d}  t={t * DT:5.1f}s\n"
            f"move={move_cmd:4.2f}  turn={turn_cmd:+5.2f}\n"
            f"heading={math.degrees(headings[t]):+6.1f}°  hit_wall={hit_wall}"
        )
        return self.trail, self.body, self.heading, self.readout


def main():
    arena = Arena(size=(100.0, 100.0), seed=0)
    runs = scenarios(arena)

    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 5.2), facecolor=SURFACE)
    panels = [AgentPanel(ax, arena, title, run) for ax, (title, run) in zip(axes, runs)]
    fig.suptitle("movement.py inside an Arena (body turns orange when hit_wall=True)",
                 fontsize=11, color=TEXT, x=0.02, ha="left")
    # leave room below the axes for the readout text, which tight_layout ignores
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.17, wspace=0.08)

    def update(t):
        return [artist for panel in panels for artist in panel.update(t)]

    anim = FuncAnimation(fig, update, frames=N_STEPS + 1, blit=True)
    os.makedirs("figures", exist_ok=True)
    out = os.path.join("figures", "movement.mp4")
    anim.save(out, writer=FFMpegWriter(fps=FPS), dpi=120, savefig_kwargs={"facecolor": SURFACE})
    print(f"saved {out}")


if __name__ == "__main__":
    main()
