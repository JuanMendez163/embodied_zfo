"""
Does the environment behave the same at different step sizes? (the to-do in docs/PORTING_NOTES.md)
Run with: python compare_dt.py  (saves figures/dt_check.png)

The same scenario is run at every dt in DTS over the same simulated time, and each run is compared with
the run at the first dt (the default). Three scenarios:
 1. open loop:      the fish follows a fixed schedule of commands (rest, cruise, sprint, hard turn), with
                    no food. Nothing the fish does depends on dt, so any difference between the runs comes
                    from the step-by-step arithmetic itself. Expected: tiny, and shrinking as dt shrinks.
 2. forager, clock: animate_env's scripted forager, but it only picks a new command every DECISION_DT
                    seconds of simulated time and keeps it until then. All runs therefore make the same
                    decisions at the same moments; only the physics between decisions is computed more
                    finely. Expected: identical, except that a pellet can be caught slightly earlier at a
                    smaller dt (eating is checked once per step), which shifts one meal.
 3. forager, every step: the same forager picking a new command at every step, as animate_env runs it.
                    At a smaller dt it decides more often, reacts sooner and swims a different route.
                    Expected: the runs drift apart. That comes from how often the policy is asked, not
                    from the environment; it is included to show how large the effect is.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np

from animate_env import AGENT, DURATION, EAT_MARK, FOOD, GRID, SEED, SURFACE, TEXT, TEXT_MUTED, TURN, scripted_action
from env import ZebrafishEnv

DTS = [0.1, 0.05, 0.02, 0.01]          # s; step sizes compared, the first is the reference
DECISION_DT = 0.1                       # s; the clocked forager re-decides on this interval
DIVERGENCE = 0.5                        # glucose units; a run has diverged once it differs from the reference by more
DT_COLORS = {0.1: AGENT, 0.05: TURN, 0.02: FOOD, 0.01: "#8e44ad"}

OPEN_LOOP_PHASES = [                    # (name, start s, end s, (move_cmd, turn_cmd, eat_cmd))
    ("rest", 0.0, 8.0, (0.0, 0.0, 0.0)),
    ("cruise", 8.0, 16.0, (0.5, 0.15, 0.0)),
    ("sprint", 16.0, 24.0, (1.0, 0.15, 0.0)),
    ("hard turn", 24.0, 32.0, (0.5, 1.0, 0.0)),
    ("rest", 32.0, 40.0, (0.0, 0.0, 0.0)),
]
OPEN_LOOP_DURATION = OPEN_LOOP_PHASES[-1][2]
PHASE_SHADES = ["#f1f0ec", "#fcfcfb"]


def open_loop_action(env, t):
    """
    The fixed-command schedule: the command of the phase containing simulated time t.
    """
    for _, start, end, command in OPEN_LOOP_PHASES:
        if start <= t < end:
            return list(command)
    return list(OPEN_LOOP_PHASES[-1][3])


def run(dt, policy, duration, decision_dt=None, **env_kwargs):
    """
    Roll out policy(env, t) in a ZebrafishEnv at step size dt for `duration` seconds of simulated time.
    decision_dt: if given, the policy is only asked every decision_dt seconds (a decision clock) and its
    command is held in between; if None it is asked at every step.
    Returns a dict of per-step arrays (index 0 = start): "t", "glucose", "position", "eaten".
    """
    env = ZebrafishEnv(dt=dt, episode_duration=duration, **env_kwargs)
    env.reset(seed=SEED)
    env.position = np.array(env.arena.size) / 2       # same start for every dt: arena centre, facing +x
    env.heading = 0.0

    times, glucose, positions, eaten = [0.0], [env.glucose], [env.position.copy()], [False]
    action, next_decision = None, 0.0
    for _ in range(env.max_steps):
        t = env.step_count * dt
        if decision_dt is None or t >= next_decision - 1e-9:
            action = policy(env, t)
            next_decision += decision_dt if decision_dt is not None else 0.0
        _, _, terminated, truncated, info = env.step(action)
        times.append(info["time"])
        glucose.append(info["glucose"])
        positions.append(env.position.copy())
        eaten.append(info["n_eaten"] > 0)
        if terminated or truncated:
            break
    return {"t": np.array(times), "glucose": np.array(glucose), "position": np.array(positions),
            "eaten": np.array(eaten)}


def on_common_grid(record, dt, grid_dt=DTS[0]):
    """
    The record sampled at multiples of grid_dt (the reference step), so runs at different dt line up.
    """
    stride = round(grid_dt / dt)
    return {k: v[::stride] for k, v in record.items()}


def compare(records):
    """
    Per dt, the differences from the reference run on the common time grid:
    max and final |glucose difference|, max position error, number of eats, and the time at which the
    glucose difference first exceeds DIVERGENCE (None if it never does).
    """
    reference = on_common_grid(records[DTS[0]], DTS[0])
    out = {}
    for dt, record in records.items():
        sampled = on_common_grid(record, dt)
        n = min(len(sampled["t"]), len(reference["t"]))      # a run that starved early is shorter
        g_err = np.abs(sampled["glucose"][:n] - reference["glucose"][:n])
        p_err = np.linalg.norm(sampled["position"][:n] - reference["position"][:n], axis=1)
        diverged = np.flatnonzero(g_err > DIVERGENCE)
        out[dt] = {
            "t": reference["t"][:n], "g_err": g_err, "p_err": p_err,
            "max_g_err": g_err.max(), "final_g_err": g_err[-1], "max_p_err": p_err.max(),
            "eats": int(record["eaten"].sum()),
            "diverged_at": float(reference["t"][diverged[0]]) if len(diverged) else None,
        }
    return out


def style(ax, duration):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, duration)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)


def plot_scenario(ax_trace, ax_err, records, stats, duration, title, note):
    """
    Left: glucose traces of every dt (eat marks on the reference). Right: |glucose - reference| over time,
    with the per-dt summary numbers as text.
    """
    for dt, record in records.items():
        ax_trace.plot(record["t"], record["glucose"], color=DT_COLORS[dt], linewidth=2.2 if dt == DTS[0] else 1.2,
                      alpha=0.9 if dt == DTS[0] else 0.8, label=f"dt = {dt}")
    ref = records[DTS[0]]
    if ref["eaten"].any():
        ax_trace.scatter(ref["t"][ref["eaten"]], ref["glucose"][ref["eaten"]], s=28, marker="^", color=EAT_MARK,
                         edgecolor=SURFACE, linewidth=0.8, zorder=4, label=f"EAT (dt = {DTS[0]})")
    ax_trace.axhline(50.0, color=TEXT_MUTED, linestyle="--", linewidth=1)
    style(ax_trace, duration)
    ax_trace.set_ylim(0, 80)
    ax_trace.set_ylabel("glucose", fontsize=9, color=TEXT_MUTED)
    ax_trace.set_title(title, fontsize=10, color=TEXT, loc="left")
    ax_trace.legend(loc="lower left", frameon=False, fontsize=8, ncol=3, labelcolor=TEXT_MUTED)

    lines = []
    for dt in DTS[1:]:
        s = stats[dt]
        ax_err.plot(s["t"], s["g_err"], color=DT_COLORS[dt], linewidth=1.4, label=f"dt = {dt}")
        diverged = f"diverges at {s['diverged_at']:.1f} s" if s["diverged_at"] is not None else "never diverges"
        lines.append(f"dt = {dt}: max |Δglucose| {s['max_g_err']:.3f}, final {s['final_g_err']:.3f}, "
                     f"max |Δposition| {s['max_p_err']:.2f}, eats {s['eats']}, {diverged}")
    ax_err.axhline(DIVERGENCE, color=TEXT_MUTED, linestyle=":", linewidth=1)
    style(ax_err, duration)
    ax_err.set_ylim(bottom=0)
    ax_err.set_ylabel(f"|glucose − glucose at dt = {DTS[0]}|", fontsize=9, color=TEXT_MUTED)
    ax_err.set_title(note, fontsize=9, color=TEXT_MUTED, loc="left")
    ax_err.text(0.01, 0.98, "\n".join(lines), transform=ax_err.transAxes, va="top", fontsize=7.5,
                color=TEXT, family="monospace", zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor="none", alpha=0.92))


def main():
    forager_kwargs = dict(arena_size=(60.0, 60.0), n_food=15, eat_radius=3.0)     # as animate_env.simulate
    scenarios = [
        ("open loop: fixed commands, no food",
         {dt: run(dt, open_loop_action, OPEN_LOOP_DURATION, n_food=0) for dt in DTS},
         OPEN_LOOP_DURATION,
         "differences here are purely numerical (cost as a lump after the exact decay; turn-then-move)"),
        (f"forager deciding every {DECISION_DT} s (decision clock)",
         {dt: run(dt, scripted_action, DURATION, decision_dt=DECISION_DT, **forager_kwargs) for dt in DTS},
         DURATION,
         "same decisions at the same simulated times; only eat timing (checked per step) can differ"),
        ("forager deciding every step",
         {dt: run(dt, scripted_action, DURATION, **forager_kwargs) for dt in DTS},
         DURATION,
         "the policy reacts at different simulated times for different dt: divergence expected"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.15, 1]})
    fig.suptitle(f"ZebrafishEnv step-size check: the same simulated time at dt = {DTS} (reference dt = {DTS[0]})",
                 fontsize=11, color=TEXT, x=0.02, ha="left")
    print(f"reference dt = {DTS[0]}; differences are against it on the common {DTS[0]} s grid")
    for row, (title, records, duration, note) in enumerate(scenarios):
        stats = compare(records)
        plot_scenario(axes[row, 0], axes[row, 1], records, stats, duration, title, note)
        print(f"\n{title}")
        for dt in DTS[1:]:
            s = stats[dt]
            diverged = f"diverges at {s['diverged_at']:.1f} s" if s["diverged_at"] is not None else "never diverges"
            print(f"  dt = {dt:<5}: max |dglucose| {s['max_g_err']:.4f}, final {s['final_g_err']:.4f}, "
                  f"max |dposition| {s['max_p_err']:.3f}, eats {s['eats']} (reference {stats[DTS[0]]['eats']}), {diverged}")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", fontsize=9, color=TEXT_MUTED)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs("figures", exist_ok=True)
    out = os.path.join("figures", "dt_check.png")
    plt.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
