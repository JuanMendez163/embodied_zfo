"""
Video sanity check for movement.py: animate an agent moving inside an Arena.
Run with: python viz/animate_movement.py  (saves figures/movement.mp4)

Five scenarios play side by side, one step per frame:
 - circle:        constant move + turn, should loop forever on the same circle
 - wall slide:    straight into a wall at an angle, should slide along it and stop in the corner
 - swimming:      smoothly varying random commands
 - rectangle:     straight segments joined by 90° turns, should retrace the same rectangle
 - rectangle-ish: a closed-loop fish steering toward the rectangle's corners, with rounded
                  corners and noise, so each lap differs slightly
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection

from viz.plot_movement import smooth_random_commands
from viz.style import FIGURES_DIR
from zfo.arena import Arena
from zfo.movement import step_pose, wrap_angle

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
    Step the agent inside the arena and record every step.
    commands is either a precomputed sequence of (move_cmd, turn_cmd) pairs (open loop), or a
    policy(position, heading) -> (move_cmd, turn_cmd) called for N_STEPS steps (closed loop).
    Returns a dict of arrays, one entry per step (index 0 is the start pose).
    """
    if callable(commands):
        policy, n_steps = commands, N_STEPS
    else:
        sequence = iter(commands)
        policy, n_steps = (lambda _position, _heading: next(sequence)), len(commands)

    positions, headings, hits = [np.array(position, dtype=float)], [heading], [False]
    taken = [(0.0, 0.0)]  # no command produced the start pose
    for _ in range(n_steps):
        move_cmd, turn_cmd = policy(positions[-1], headings[-1])
        pos, heading, hit_wall = step_pose(positions[-1], headings[-1], move_cmd, turn_cmd,
                                           MAX_SPEED, MAX_TURN, DT, arena.size, BODY_RADIUS)
        positions.append(pos)
        headings.append(heading)
        hits.append(hit_wall)
        taken.append((move_cmd, turn_cmd))
    commands = taken
    return {
        "positions": np.array(positions),
        "headings": np.array(headings),
        "hits": np.array(hits),
        "commands": np.array(commands),
    }


def rectangle_commands(n_steps, long_side=40, short_side=20):
    """
    (move_cmd, turn_cmd) pairs that trace a rectangle counterclockwise, repeating until n_steps:
    straight for long_side units, turn 90° left in place, straight for short_side units, turn 90° left, ...
    Each straight step covers MAX_SPEED * DT units; a full-rate turn step rotates MAX_TURN * DT rad.
    Step counts are truncated with int(), so sides and corners are exact only when these divide evenly.
    """
    long_steps = int(long_side / (MAX_SPEED * DT))
    short_steps = int(short_side / (MAX_SPEED * DT))
    turn_steps = int((math.pi / 2) / (MAX_TURN * DT))  # 90° left turn
    commands = []
    while len(commands) < n_steps:
        commands.extend([(1.0, 0.0)] * long_steps)   # straight
        commands.extend([(0.0, 1.0)] * turn_steps)   # turn left
        commands.extend([(1.0, 0.0)] * short_steps)  # straight
        commands.extend([(0.0, 1.0)] * turn_steps)   # turn left
    return commands[:n_steps]


class WaypointFish:
    """
    Closed-loop policy that swims a loop through waypoints the way a fish might:
    it steers toward the current waypoint from wherever it actually is, turns while moving
    (rounded corners), and adds smoothly varying noise to its commands.
    Call it as policy(position, heading) -> (move_cmd, turn_cmd).
    """

    SPEED_MODES = ("coupled", "cruise", "cosine")

    def __init__(self, waypoints, rng, arrive_radius=4.0, noise=0.15, smoothing=0.9,
                 speed_mode="coupled", cruise_speed=0.3):
        if speed_mode not in self.SPEED_MODES:
            raise ValueError(f"speed_mode must be one of {self.SPEED_MODES}, got {speed_mode!r}")
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.rng = rng
        self.arrive_radius = arrive_radius  # switch to the next waypoint once this close
        self.noise = noise                  # size of the command noise
        self.smoothing = smoothing          # noise persistence, as in smooth_random_commands
        self.speed_mode = speed_mode        # how forward speed drops while turning (see steer)
        self.cruise_speed = cruise_speed    # "cruise" mode: minimum move_cmd, even in the sharpest turn
        self.target = 0
        self.corners_reached = 0  # total waypoint arrivals, across laps
        self.noise_state = np.zeros(2)

    def steer(self, heading_error):
        """
        Map heading_error (radians in [-pi, pi], the signed angle from the agent's heading to the
        direction of the target; positive = target is to the left) to noise-free (move_cmd, turn_cmd).
        speed_mode sets how move_cmd slows for turns:
         - "coupled": 1 - |turn_cmd|; stops and pivots when the target is >= 90° away
         - "cruise":  same shape, but never below cruise_speed, so corners are swept rather than pivoted
         - "cosine":  cos(heading_error), set by the error itself rather than by turn_cmd, so
                      changing the turning gain doesn't change the speed profile
        """
        # turn_cmd is proportional to heading_error, but clipped to [-1, 1]
        turn_cmd = np.clip(heading_error / (math.pi / 2), -1.0, 1.0)
        # move_cmd is 1.0 when heading_error is small, and decreases as heading_error grows
        if self.speed_mode == "coupled":
            move_cmd = 1.0 - abs(turn_cmd)
        elif self.speed_mode == "cruise":
            move_cmd = self.cruise_speed + (1.0 - self.cruise_speed) * (1.0 - abs(turn_cmd))
        else:  # "cosine"
            move_cmd = max(0.0, math.cos(heading_error))
        return move_cmd, turn_cmd

    def __call__(self, position, heading):
        if np.linalg.norm(self.waypoints[self.target] - position) < self.arrive_radius:
            self.target = (self.target + 1) % len(self.waypoints)
            self.corners_reached += 1

        to_target = self.waypoints[self.target] - position
        heading_error = wrap_angle(math.atan2(to_target[1], to_target[0]) - heading)
        move_cmd, turn_cmd = self.steer(heading_error)

        self.noise_state = (self.smoothing * self.noise_state
                            + (1 - self.smoothing) * self.rng.normal(0, self.noise / (1 - self.smoothing), size=2))
        return (float(np.clip(move_cmd + self.noise_state[0], 0.0, 1.0)),
                float(np.clip(turn_cmd + self.noise_state[1], -1.0, 1.0)))


RECTANGLE_CORNERS = [[70, 35], [70, 55], [30, 55], [30, 35]]  # counterclockwise, same rectangle as above


def scenarios(arena):
    return [
        ("Circle (move=1, turn=0.2)",
         rollout(arena, [50, 30], 0.0, [(1.0, 0.2)] * N_STEPS)),
        ("Wall slide (move=1, turn=0, heading 60°)",
         rollout(arena, [70, 15], math.radians(60), [(1.0, 0.0)] * N_STEPS)),
        ("Swimming (smooth random commands)",
         rollout(arena, arena.random_position(margin=20), 0.0,
                 smooth_random_commands(arena.np_random, N_STEPS))),
        ("Rectangle (straight + 90° turns)",
         rollout(arena, [30, 35], 0.0, rectangle_commands(N_STEPS))),
        ("Rectangle-ish (fish steering to corners)",
         {**rollout(arena, [30, 35], 0.0, WaypointFish(RECTANGLE_CORNERS, np.random.default_rng(1))),
          "waypoints": np.array(RECTANGLE_CORNERS)}),
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
        if "waypoints" in run:
            ax.scatter(*run["waypoints"].T, marker="x", s=30, color=TEXT_MUTED, zorder=1)

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
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "movement.mp4")
    anim.save(out, writer=FFMpegWriter(fps=FPS), dpi=120, savefig_kwargs={"facecolor": SURFACE})
    print(f"saved {out}")


if __name__ == "__main__":
    main()
