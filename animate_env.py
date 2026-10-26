"""
Video check of ZebrafishEnv's glucose dynamics: the agent moves in the arena (left) while a live
glucose trace (right, top) and its commands (right, bottom) update.
Run with: python animate_env.py  (saves figures/env_glucose.mp4)

A scripted policy runs through phases, shaded on the traces:
 - rest:    move 0                          -> basal decay only
 - cruise:  move 0.5, gentle turn           -> small extra cost (cost grows with move_cmd**2)
 - sprint:  move 1.0, gentle turn           -> full movement cost, steepest decline
 - forage:  steer to the nearest pellet; EAT only while below the glucose target
            -> +intake jumps, cooldown between eats, food ignored once sated

The policy reads env state directly (food positions, glucose); that is fine for a demo, while the
RL agent will only see the observation.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Wedge

from env import ZebrafishEnv
from movement import wrap_angle

PHASES = [("rest", 0.0, 8.0), ("cruise", 8.0, 16.0), ("sprint", 16.0, 24.0), ("forage", 24.0, 70.0)]
DURATION = PHASES[-1][2]
FPS = 30              # 3x real time at dt = 0.1
TRAIL = 50
SEED = 2

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
AGENT = "#2a78d6"
FOOD = "#1baf7a"
TURN = "#eb6834"
EAT_MARK = "#008300"
PHASE_SHADES = ["#f1f0ec", "#fcfcfb", "#f1f0ec", "#fcfcfb"]


def phase_at(t):
    for name, start, end in PHASES:
        if start <= t < end:
            return name
    return PHASES[-1][0]


def scripted_action(env, t):
    """
    The demo policy: fixed commands in the first three phases, then hunger-gated foraging.
    """
    phase = phase_at(t)
    if phase == "rest":
        return [0.0, 0.0, 0.0]
    if phase == "cruise":
        return [0.5, 0.15, 0.0]
    if phase == "sprint":
        return [1.0, 0.15, 0.0]
    return forage_action(env)


def forage_action(env):
    """
    Hunger-gated foraging: steer to the nearest pellet and EAT while below the glucose target,
    otherwise cruise slowly and ignore food. Also the scripted baseline in train_ppo.py.
    """
    hungry = env.glucose < env.glucose_target
    food = env.arena.food_positions
    if not hungry or len(food) == 0:
        return [0.3, 0.1, 0.0]                      # sated: slow cruise, ignore food
    offsets = food - env.position
    distances = np.linalg.norm(offsets, axis=1)
    target = offsets[np.argmin(distances)]
    heading_error = wrap_angle(math.atan2(target[1], target[0]) - env.heading)
    turn_cmd = float(np.clip(heading_error / (math.pi / 2), -1.0, 1.0))
    move_cmd = 1.0 - abs(turn_cmd)
    # pivot in place when a nearby pellet is outside the mouth cone: any forward speed there makes the
    # agent orbit the pellet (turning radius ~ distance to it), keeping it beside the mouth forever
    if distances.min() < 2 * env.eat_radius and abs(heading_error) > env.mouth_angle / 2:
        move_cmd = 0.0
    return [move_cmd, turn_cmd, 1.0]                # hungry: try to eat whenever possible


def simulate():
    env = ZebrafishEnv(arena_size=(60.0, 60.0), n_food=15, eat_radius=3.0, episode_duration=DURATION)
    env.reset(seed=SEED)
    env.position = np.array([30.0, 30.0])           # start centered so the fixed-command circles fit
    env.heading = 0.0

    frames = [{"t": 0.0, "position": env.position.copy(), "heading": env.heading, "glucose": env.glucose,
               "food": env.arena.food_positions.copy(), "action": [0.0, 0.0, 0.0], "eaten": False}]
    while True:
        t = frames[-1]["t"]
        action = scripted_action(env, t)
        _, _, terminated, truncated, info = env.step(action)
        frames.append({"t": info["time"], "position": env.position.copy(), "heading": env.heading,
                       "glucose": info["glucose"], "food": env.arena.food_positions.copy(),
                       "action": action, "eaten": info["n_eaten"] > 0})
        if terminated or truncated:
            break
    return env, frames


def shade_phases(ax, label=False):
    for (name, start, end), shade in zip(PHASES, PHASE_SHADES):
        ax.axvspan(start, end, color=shade, zorder=0, linewidth=0)
        if label:
            ax.text((start + end) / 2, 1.02, name, transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=9, color=TEXT_MUTED)


def style_trace(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, DURATION)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)


def main():
    env, frames = simulate()
    times = np.array([f["t"] for f in frames])
    glucose = np.array([f["glucose"] for f in frames])
    actions = np.array([f["action"] for f in frames])
    eat_times = times[[f["eaten"] for f in frames]]
    eat_glucose = glucose[[f["eaten"] for f in frames]]

    fig = plt.figure(figsize=(13, 6.2), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[2, 1],
                            left=0.03, right=0.94, top=0.86, bottom=0.12, wspace=0.12, hspace=0.45)

    # --- arena ---
    ax_arena = fig.add_subplot(grid[:, 0])
    ax_arena.set_facecolor(SURFACE)
    env.arena.reset()                                # walls only; food drawn per frame
    env.arena.draw(ax_arena, edgecolor=TEXT_MUTED)
    ax_arena.set_xticks([])
    ax_arena.set_yticks([])
    for spine in ax_arena.spines.values():
        spine.set_visible(False)
    food = ax_arena.scatter([], [], s=30, color=FOOD, edgecolor="none", zorder=2)
    trail = LineCollection([], linewidths=1.5, color=AGENT, alpha=0.35)
    ax_arena.add_collection(trail)
    cone = Wedge((0, 0), env.eat_radius, 0, 0, facecolor=AGENT, edgecolor=AGENT, alpha=0.2, zorder=3)
    body = Circle((0, 0), env.body_radius, color=AGENT, zorder=4)
    ax_arena.add_patch(cone)
    ax_arena.add_patch(body)
    (heading_line,) = ax_arena.plot([], [], color=TEXT, linewidth=1.5, zorder=5)
    status = ax_arena.text(0.0, -0.02, "", transform=ax_arena.transAxes, va="top", fontsize=9,
                           color=TEXT_MUTED, family="monospace")

    # --- glucose trace ---
    ax_g = fig.add_subplot(grid[0, 1])
    style_trace(ax_g)
    shade_phases(ax_g, label=True)
    ax_g.set_ylim(0, env.glucose_max * 0.8)
    ax_g.axhline(env.glucose_target, color=TEXT_MUTED, linestyle="--", linewidth=1)
    ax_g.text(DURATION, env.glucose_target, " target", va="center", fontsize=8, color=TEXT_MUTED)
    ax_g.axhline(env.starvation_threshold, color=TURN, linestyle=":", linewidth=1)
    ax_g.text(DURATION, env.starvation_threshold, " starve", va="center", fontsize=8, color=TEXT_MUTED)
    ax_g.set_ylabel("glucose", fontsize=9, color=TEXT_MUTED)
    (glucose_line,) = ax_g.plot([], [], color=AGENT, linewidth=2)
    eat_marks = ax_g.scatter([], [], s=40, marker="^", color=EAT_MARK, edgecolor=SURFACE, linewidth=1,
                             zorder=4, label="successful EAT")
    ax_g.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=TEXT_MUTED)
    cursor_g = ax_g.axvline(0, color=TEXT_MUTED, linewidth=0.8)

    # --- command trace ---
    ax_c = fig.add_subplot(grid[1, 1], sharex=ax_g)
    style_trace(ax_c)
    shade_phases(ax_c)
    ax_c.set_ylim(-0.05, 1.15)
    ax_c.set_xlabel("time (s)", fontsize=9, color=TEXT_MUTED)
    ax_c.set_ylabel("command", fontsize=9, color=TEXT_MUTED)
    (move_line,) = ax_c.plot([], [], color=AGENT, linewidth=1.5, label="move_cmd")
    (turn_line,) = ax_c.plot([], [], color=TURN, linewidth=1.5, label="|turn_cmd|")
    ax_c.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8, ncol=2,
                labelcolor=TEXT_MUTED)
    cursor_c = ax_c.axvline(0, color=TEXT_MUTED, linewidth=0.8)

    fig.suptitle("ZebrafishEnv: glucose under rest, cruising, sprinting and foraging (placeholder rates)",
                 fontsize=11, color=TEXT, x=0.03, ha="left")

    def update(i):
        frame = frames[i]
        x, y = frame["position"]
        heading_deg = math.degrees(frame["heading"])
        half_mouth = math.degrees(env.mouth_angle) / 2

        food.set_offsets(frame["food"] if len(frame["food"]) else np.zeros((0, 2)))
        points = np.array([f["position"] for f in frames[max(0, i - TRAIL):i + 1]])
        trail.set_segments(np.stack([points[:-1], points[1:]], axis=1) if len(points) > 1 else [])
        cone.set_center((x, y))
        cone.set_theta1(heading_deg - half_mouth)
        cone.set_theta2(heading_deg + half_mouth)
        cone.set_alpha(0.6 if frame["eaten"] else 0.2)
        body.center = (x, y)
        tip = frame["position"] + 2.5 * env.body_radius * np.array([math.cos(frame["heading"]),
                                                                    math.sin(frame["heading"])])
        heading_line.set_data([x, tip[0]], [y, tip[1]])
        move_cmd, turn_cmd, eat_cmd = frame["action"]
        status.set_text(f"t={frame['t']:5.1f}s  phase={phase_at(frame['t']):<7s} glucose={frame['glucose']:5.1f}\n"
                        f"move={move_cmd:4.2f}  turn={turn_cmd:+5.2f}  eat={'yes' if eat_cmd > 0.5 else 'no '}"
                        f"  food left={len(frame['food'])}")

        glucose_line.set_data(times[:i + 1], glucose[:i + 1])
        shown = eat_times <= frame["t"]
        eat_marks.set_offsets(np.column_stack([eat_times[shown], eat_glucose[shown]]) if shown.any()
                              else np.zeros((0, 2)))
        move_line.set_data(times[1:i + 1], actions[1:i + 1, 0])
        turn_line.set_data(times[1:i + 1], np.abs(actions[1:i + 1, 1]))
        cursor_g.set_xdata([frame["t"], frame["t"]])
        cursor_c.set_xdata([frame["t"], frame["t"]])
        return (food, trail, cone, body, heading_line, status, glucose_line, eat_marks, move_line, turn_line,
                cursor_g, cursor_c)

    anim = FuncAnimation(fig, update, frames=len(frames), blit=True)
    os.makedirs("figures", exist_ok=True)
    out = os.path.join("figures", "env_glucose.mp4")
    anim.save(out, writer=FFMpegWriter(fps=FPS), dpi=110, savefig_kwargs={"facecolor": SURFACE})
    print(f"saved {out}: {len(frames) - 1} steps, {len(eat_times)} eats, final glucose {glucose[-1]:.1f}")


if __name__ == "__main__":
    main()
