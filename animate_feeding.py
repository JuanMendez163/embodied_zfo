"""
Video sanity check for feeding.py: a scripted forager swims to food and eats it with an explicit
EAT action, with its mouth cone drawn.
Run with: python animate_feeding.py  (saves figures/feeding.mp4)

The forager is a stand-in for the future RL policy: each step it steers toward the nearest
pellet and chooses EAT whenever a pellet is in its mouth cone. Within a step it eats first (at
its current pose), then moves, as in wef's EFishAgent.step.

Pellet colors:
 - edible:           inside eat_radius and inside the mouth cone (EAT would take the nearest one)
 - in reach, behind: inside eat_radius but outside the cone, so EAT cannot take it
 - out of reach:     everything else
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Wedge

from arena import Arena
from feeding import food_in_reach, nearest_food_in_reach
from movement import step_pose, wrap_angle

ARENA_SIZE = (50.0, 50.0)
N_FOOD = 15
MAX_SPEED = 10.0              # units/s
MAX_TURN = math.pi            # rad/s
DT = 0.1                      # s
BODY_RADIUS = 1.0
EAT_RADIUS = 4.0
MOUTH_ANGLE = math.radians(90)
N_STEPS = 175                 # with SEED = 3 the forager has eaten everything by step ~155
FPS = 20                      # 2x real time
TRAIL = 40
FLASH_STEPS = 4               # frames the cone stays highlighted after a successful EAT
SEED = 3

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
AGENT = "#2a78d6"
EDIBLE = "#008300"            # green
IN_REACH_BEHIND = "#eb6834"   # orange
OUT_OF_REACH = "#9fd8c2"      # pale aqua
CONE = "#2a78d6"


def forager_action(position, heading, food_positions):
    """
    Scripted policy: steer toward the nearest pellet, slow down for sharp turns, and EAT whenever
    something is in the mouth cone. Returns (move_cmd, turn_cmd, eat).
    """
    eat = nearest_food_in_reach(position, heading, food_positions, EAT_RADIUS, MOUTH_ANGLE) is not None
    if len(food_positions) == 0:
        return 0.0, 0.0, False
    offsets = food_positions - position
    target = offsets[np.argmin(np.linalg.norm(offsets, axis=1))]
    heading_error = wrap_angle(math.atan2(target[1], target[0]) - heading)
    turn_cmd = float(np.clip(heading_error / (math.pi / 2), -1.0, 1.0))
    move_cmd = 0.2 + 0.8 * (1.0 - abs(turn_cmd))
    return move_cmd, turn_cmd, eat


def simulate(arena):
    """
    Run the forager and record, for every step, the pose, the food present, and what was eaten.
    """
    position = arena.random_position(margin=5.0)
    heading = float(arena.np_random.uniform(-math.pi, math.pi))
    arena.spawn_food(avoid=position)

    frames = []
    eaten_total = 0
    for _ in range(N_STEPS + 1):
        move_cmd, turn_cmd, eat = forager_action(position, heading, arena.food_positions)
        frames.append({
            "position": position.copy(),
            "heading": heading,
            "food": arena.food_positions.copy(),
            "edible": food_in_reach(position, heading, arena.food_positions, EAT_RADIUS, MOUTH_ANGLE),
            "in_reach": food_in_reach(position, heading, arena.food_positions, EAT_RADIUS),
            "eat": eat,
            "eaten_total": eaten_total,
            "move_cmd": move_cmd,
            "turn_cmd": turn_cmd,
        })

        # eat first, at the current pose ...
        if eat:
            arena.remove_food([nearest_food_in_reach(position, heading, arena.food_positions,
                                                     EAT_RADIUS, MOUTH_ANGLE)])
            eaten_total += 1
        # ... then move
        position, heading, _ = step_pose(position, heading, move_cmd, turn_cmd, MAX_SPEED, MAX_TURN, DT,
                                         arena.size, BODY_RADIUS)
    return frames


def pellet_colors(frame):
    colors = np.array([OUT_OF_REACH] * len(frame["food"]), dtype=object)
    colors[frame["in_reach"]] = IN_REACH_BEHIND
    colors[frame["edible"]] = EDIBLE
    return list(colors)


def main():
    arena = Arena(size=ARENA_SIZE, n_food=N_FOOD, food_min_distance=8.0, seed=SEED)
    frames = simulate(arena)

    fig, ax = plt.subplots(figsize=(6.4, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    arena.reset()  # draw walls only; pellets are drawn per frame below
    arena.draw(ax, edgecolor=TEXT_MUTED)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    food = ax.scatter([], [], s=36, edgecolor="none", zorder=2)
    trail = LineCollection([], linewidths=1.5, color=AGENT, alpha=0.35)
    ax.add_collection(trail)
    reach = Circle((0, 0), EAT_RADIUS, fill=False, edgecolor=CONE, linewidth=0.8, linestyle="--", zorder=3)
    cone = Wedge((0, 0), EAT_RADIUS, 0, 0, facecolor=CONE, edgecolor=CONE, alpha=0.2, zorder=3)
    body = Circle((0, 0), BODY_RADIUS, color=AGENT, zorder=4)
    for patch in (reach, cone, body):
        ax.add_patch(patch)
    (heading_line,) = ax.plot([], [], color=TEXT, linewidth=1.5, zorder=5)
    readout = ax.text(0.0, -0.02, "", transform=ax.transAxes, va="top", fontsize=9,
                      color=TEXT_MUTED, family="monospace")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=EDIBLE, label="edible (in cone)"),
        plt.Line2D([], [], marker="o", linestyle="none", color=IN_REACH_BEHIND, label="in reach, outside cone"),
        plt.Line2D([], [], marker="o", linestyle="none", color=OUT_OF_REACH, label="out of reach"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              fontsize=8, labelcolor=TEXT_MUTED)
    fig.suptitle(f"feeding.py: eat_radius={EAT_RADIUS:g}, mouth_angle={math.degrees(MOUTH_ANGLE):.0f}°",
                 fontsize=11, color=TEXT, x=0.03, ha="left")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.12)

    last_eat = -FLASH_STEPS - 1

    def update(t):
        nonlocal last_eat
        frame = frames[t]
        x, y = frame["position"]
        heading_deg = math.degrees(frame["heading"])

        food.set_offsets(frame["food"] if len(frame["food"]) else np.zeros((0, 2)))
        food.set_facecolor(pellet_colors(frame))

        start = max(0, t - TRAIL)
        points = np.array([f["position"] for f in frames[start:t + 1]])
        trail.set_segments(np.stack([points[:-1], points[1:]], axis=1) if len(points) > 1 else [])

        if frame["eat"]:
            last_eat = t
        flashing = t - last_eat <= FLASH_STEPS
        reach.center = (x, y)
        cone.set_center((x, y))
        cone.set_theta1(heading_deg - math.degrees(MOUTH_ANGLE) / 2)
        cone.set_theta2(heading_deg + math.degrees(MOUTH_ANGLE) / 2)
        cone.set_alpha(0.6 if flashing else 0.2)
        body.center = (x, y)
        tip = frame["position"] + 2.5 * BODY_RADIUS * np.array([math.cos(frame["heading"]),
                                                                math.sin(frame["heading"])])
        heading_line.set_data([x, tip[0]], [y, tip[1]])

        readout.set_text(
            f"step {t:3d}  t={t * DT:5.1f}s   eaten {frame['eaten_total']:2d}/{N_FOOD}"
            f"   {'EAT' if frame['eat'] else '   '}\n"
            f"move={frame['move_cmd']:4.2f}  turn={frame['turn_cmd']:+5.2f}  "
            f"in reach={len(frame['in_reach'])}  edible={len(frame['edible'])}"
        )
        return food, trail, reach, cone, body, heading_line, readout

    anim = FuncAnimation(fig, update, frames=len(frames), blit=True)
    os.makedirs("figures", exist_ok=True)
    out = os.path.join("figures", "feeding.mp4")
    anim.save(out, writer=FFMpegWriter(fps=FPS), dpi=130, savefig_kwargs={"facecolor": SURFACE})
    print(f"saved {out}: ate {frames[-1]['eaten_total']}/{N_FOOD} pellets in {N_STEPS} steps")


if __name__ == "__main__":
    main()
