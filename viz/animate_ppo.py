"""
Video of a trained PPO agent (runs/NAME/model.zip) in one evaluation episode: the agent in the arena
(left), its glucose trace (right, top) and its commands (right, bottom).
Run with: python viz/animate_ppo.py [--name ppo] [--seed 10000]  (saves figures/ppo_NAME_seedSEED.mp4)

The episode is the same one training.metrics.evaluate runs for that seed (make_ppo_env, deterministic
actions), and its episode_metrics are printed. The policy only sees the observation; env state is
read for drawing only. The mouth cone is drawn darker while EAT is held and flashes on a successful eat.
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Wedge
from stable_baselines3 import PPO

from training.metrics import EVAL_SEED, episode_metrics
from training.train_ppo import RUNS_DIR, make_ppo_env
from viz.style import AGENT, EAT_MARK, FIGURES_DIR, FOOD, SURFACE, TEXT, TEXT_MUTED, TURN, style_trace

FPS = 30              # 3x real time at dt = 0.1
TRAIL = 50            # past steps shown behind the agent
CONE_ALPHA = {"idle": 0.12, "held": 0.35, "eaten": 0.8}


def simulate(model, seed):
    """
    One deterministic episode of the policy on make_ppo_env, reset with seed.
    Returns the unwrapped env, per-step frames for drawing, and the step infos (for episode_metrics).
    """
    env = make_ppo_env()
    obs, info = env.reset(seed=seed)
    base = env.unwrapped
    frames = [{"t": 0.0, "position": base.position.copy(), "heading": base.heading, "glucose": base.glucose,
               "food": base.arena.food_positions.copy(), "action": np.zeros(3), "eat_held": False,
               "eaten": False}]
    infos, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        infos.append(info)
        frames.append({"t": info["time"], "position": base.position.copy(), "heading": base.heading,
                       "glucose": info["glucose"], "food": base.arena.food_positions.copy(),
                       "action": info["action"], "eat_held": info["eat_attempted"],
                       "eaten": info["n_eaten"] > 0})
        done = terminated or truncated
    return base, frames, infos


def main(name, seed):
    model = PPO.load(os.path.join(RUNS_DIR, name, "model"), device="cpu")
    env, frames, infos = simulate(model, seed)
    metrics = episode_metrics(infos, env.dt, env.glucose_target, env.max_steps * env.dt, env.eat_cooldown)
    duration = env.max_steps * env.dt

    times = np.array([f["t"] for f in frames])
    glucose = np.array([f["glucose"] for f in frames])
    actions = np.array([f["action"] for f in frames])
    eat_held = np.array([f["eat_held"] for f in frames], dtype=float)
    eaten = np.array([f["eaten"] for f in frames])

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
    cone = Wedge((0, 0), env.eat_radius, 0, 0, facecolor=AGENT, edgecolor=AGENT, zorder=3)
    body = Circle((0, 0), env.body_radius, color=AGENT, zorder=4)
    ax_arena.add_patch(cone)
    ax_arena.add_patch(body)
    (heading_line,) = ax_arena.plot([], [], color=TEXT, linewidth=1.5, zorder=5)
    status = ax_arena.text(0.0, -0.02, "", transform=ax_arena.transAxes, va="top", fontsize=9,
                           color=TEXT_MUTED, family="monospace")

    # --- glucose trace ---
    ax_g = fig.add_subplot(grid[0, 1])
    style_trace(ax_g, duration)
    ax_g.set_ylim(0, env.glucose_max)
    ax_g.axhline(env.glucose_target, color=TEXT_MUTED, linestyle="--", linewidth=1)
    ax_g.text(duration, env.glucose_target, " target", va="center", fontsize=8, color=TEXT_MUTED)
    ax_g.axhline(env.starvation_threshold, color=TURN, linestyle=":", linewidth=1)
    ax_g.text(duration, env.starvation_threshold, " starve", va="center", fontsize=8, color=TEXT_MUTED)
    ax_g.set_ylabel("glucose", fontsize=9, color=TEXT_MUTED)
    (glucose_line,) = ax_g.plot([], [], color=AGENT, linewidth=2)
    eat_marks = ax_g.scatter([], [], s=40, marker="^", color=EAT_MARK, edgecolor=SURFACE, linewidth=1,
                             zorder=4, label="successful EAT")
    ax_g.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=TEXT_MUTED)
    cursor_g = ax_g.axvline(0, color=TEXT_MUTED, linewidth=0.8)

    # --- command trace ---
    ax_c = fig.add_subplot(grid[1, 1], sharex=ax_g)
    style_trace(ax_c, duration)
    ax_c.set_ylim(-0.05, 1.15)
    ax_c.set_xlabel("time (s)", fontsize=9, color=TEXT_MUTED)
    ax_c.set_ylabel("command", fontsize=9, color=TEXT_MUTED)
    (move_line,) = ax_c.plot([], [], color=AGENT, linewidth=1.5, label="move_cmd")
    (turn_line,) = ax_c.plot([], [], color=TURN, linewidth=1.5, label="|turn_cmd|")
    (eat_line,) = ax_c.plot([], [], color=EAT_MARK, linewidth=1, alpha=0.6, drawstyle="steps-post",
                            label="EAT held")
    ax_c.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8, ncol=3,
                labelcolor=TEXT_MUTED)
    cursor_c = ax_c.axvline(0, color=TEXT_MUTED, linewidth=0.8)

    fig.suptitle(f"PPO agent '{name}', evaluation episode seed {seed} (start glucose {glucose[0]:.0f}): "
                 f"target error {metrics['glucose_target_error']:.1f}, {metrics['total_eaten']:.0f} eaten, "
                 f"EAT held {100 * metrics['eat_attempt_rate']:.0f}% of the time",
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
        cone.set_alpha(CONE_ALPHA["eaten" if frame["eaten"] else "held" if frame["eat_held"] else "idle"])
        body.center = (x, y)
        tip = frame["position"] + 2.5 * env.body_radius * np.array([math.cos(frame["heading"]),
                                                                    math.sin(frame["heading"])])
        heading_line.set_data([x, tip[0]], [y, tip[1]])
        move_cmd, turn_cmd, _ = frame["action"]
        status.set_text(f"t={frame['t']:5.1f}s  glucose={frame['glucose']:5.1f}  food left={len(frame['food'])}\n"
                        f"move={move_cmd:4.2f}  turn={turn_cmd:+5.2f}  eat={'held' if frame['eat_held'] else '-   '}")

        glucose_line.set_data(times[:i + 1], glucose[:i + 1])
        shown = eaten[:i + 1]
        eat_marks.set_offsets(np.column_stack([times[:i + 1][shown], glucose[:i + 1][shown]]) if shown.any()
                              else np.zeros((0, 2)))
        move_line.set_data(times[1:i + 1], actions[1:i + 1, 0])
        turn_line.set_data(times[1:i + 1], np.abs(actions[1:i + 1, 1]))
        eat_line.set_data(times[1:i + 1], eat_held[1:i + 1])
        cursor_g.set_xdata([frame["t"], frame["t"]])
        cursor_c.set_xdata([frame["t"], frame["t"]])
        return (food, trail, cone, body, heading_line, status, glucose_line, eat_marks, move_line, turn_line,
                eat_line, cursor_g, cursor_c)

    anim = FuncAnimation(fig, update, frames=len(frames), blit=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, f"ppo_{name}_seed{seed}.mp4")
    anim.save(out, writer=FFMpegWriter(fps=FPS), dpi=110, savefig_kwargs={"facecolor": SURFACE})
    print(f"saved {out}: {len(frames) - 1} steps")
    print("  " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ppo")
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    args = parser.parse_args()
    main(args.name, args.seed)
