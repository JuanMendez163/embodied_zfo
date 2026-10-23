# Porting notes

Status, decisions and next steps for porting embodied_organs' `GridWorld` to a continuous
environment. Update this file whenever a step is completed or a decision is made.

_Last updated: 2026-10-23_

## Status

| Step | State | Where |
|---|---|---|
| 1. Movement | Done | `movement.py`, `test_movement.py`, `plot_movement.py`, `animate_movement.py`, `compare_steering.py` |
| 2a. Food placement | Done | `arena.py`, `test_arena.py`, `plot_food.py` |
| 2b. Eating rule | Done | `feeding.py`, `test_feeding.py`, `animate_feeding.py` |
| 3a. Glucose dynamics (was 2c) | Done | `metabolism.py`, `test_metabolism.py` |
| 3b.1 Gymnasium env dynamics | Done | `env.py`, `test_env.py`, `animate_env.py` |
| **3b.2 Observation** | **Next** | `env.py` (`obs_mode`) |
| 3c.1 dt-independence check of the full env | Done | `compare_dt.py`, `test_env.py` (`test_full_env_roughly_independent_of_dt`) |
| 3c.2 RL training | Not started | |

60 tests pass (`pytest -q`).

## Next step: 3b.2 observation

Decision: **option C**, both sensory modes behind `obs_mode`, starting with A.
Currently `obs_mode="internal"` returns only `[glucose / glucose_max, can_eat]` as a placeholder.

- **A. Nearest k pellets (do first):** for each of the k nearest pellets, egocentric distance and
  direction (`cos`, `sin` of the bearing relative to heading), padded/masked when fewer than k remain.
  Easy to learn from; less biological (no range limit).
- **B. Sector sensors (wef style):** split the field of view into S angular sectors, each reporting
  the proximity `1 - d / range` of the nearest pellet inside a sensing range. Reference:
  `EFishAgent._update_sensors` in `wef/onpolicy/custom/fish/MAEFish.py`.
- Both include internal state (normalized glucose, can_eat). Observation must be a fixed-shape
  `Box`; keep `check_env` passing.

## Design decisions (and why)

- **Eating is an explicit action** (`eat_cmd > 0.5`), as in GridWorld's EAT, rather than automatic on
  contact as in wef. More ethologically realistic: the agent chooses when to eat instead of steering
  away from food to avoid overeating.
- **One pellet per EAT**, the nearest in reach (GridWorld: one `intake_amount` per EAT).
  `update_glucose` still accepts `n_eaten > 1` so metabolism stays independent of the eating rule.
- **Mouth cone:** a pellet must be within `eat_radius` and within `mouth_angle / 2` of the heading
  (full-width convention, as in wef's `eating_angle`). Defaults: `eat_radius = 2`, `mouth_angle = 90°`.
- **Eating cooldown in seconds** (`EAT_COOLDOWN = 0.5 s`, placeholder handling time) so the maximum
  eating rate does not depend on `dt`. **Failed EAT attempts do not restart the cooldown** (only
  successful eats count); deliberate, simpler for early learning.
- **Eat and move in the same step**, with eat first (at the pose the agent observed), then move.
  More realistic than GridWorld's either/or at `dt = 0.1 s` (a capture strike includes forward motion).
- **Glucose dynamics:** exact exponential basal decay `g * exp(-k dt)` (k from a half-life), plus
  movement cost `(c_m * move^p + c_turn * |turn|^p) * dt`, plus intake, clamped to `[0, glucose_max]`,
  snapped to 0 below the starvation threshold. Turning costs glucose (otherwise RL agents spin for free).
- **Reward** = reduction in drive (Gutkin / GridWorld); no separate starvation penalty.
  `terminated` = starved, `truncated` = time limit (episode length in seconds).
- **First-order movement** (commands set velocities directly), turn-then-move within a step (as in
  wef). Walls clip each axis (agent slides). Second-order dynamics (momentum, drag) not ported yet.

## Open questions / to do

- **Literature values** for the placeholders in `metabolism.py` (glucose half-life, movement and
  turning cost rates, cost exponent; drag-based power suggests p ~ 3), `INTAKE_AMOUNT` (glucose per
  prey item; current +10 causes a ~5-7 overshoot sawtooth around the target), and `EAT_COOLDOWN`.
  Changing the half-life shifts the balance with movement cost; revisit both together.
- **Speed-dependent eating** (option, not built): success probability `1 - move_cmd` (wef's
  `probabilistic_eating`) or a speed threshold. Open sub-question: use this step's `move_cmd` or the
  previous step's (the latter needs the last action in the observation).
- **Handling freeze** (wef stops movement during the eat cooldown): optional, not built.
- **Second-order movement** (`motion_order=2` in wef) not ported.
- **dt-independence (checked 2026-10-23, `compare_dt.py`):** with fixed commands over 40 s and no food,
  glucose at `dt = 0.05 / 0.02 / 0.01` differs from `dt = 0.1` by at most 0.014 and position by at most
  0.6 units (during a hard turn). Both are O(dt) and come from (i) the movement cost being subtracted as a
  lump after the exact decay, and (ii) turn-then-move kinematics. Either could be made exact for
  piecewise-constant commands (`dg/dt = -k g - c` has the closed form `g e^{-k dt} - (c / k)(1 - e^{-k dt})`;
  constant speed and turn rate trace an exact circular arc), but the residual is far below anything the
  agent could notice, so neither was changed. The closed-loop forager is a different matter: eating is an
  event checked once per env step, so at a finer dt a pellet can be caught up to one coarse step earlier,
  which shifts one meal and, through the path, the rest of the episode (same eat count, final glucose within
  ~0.6 for a forager re-deciding every 0.1 s). A policy that re-decides every env step diverges far more
  (paths differ by tens of units at `dt <= 0.02`) because it reacts at different simulated times.
- **Decision interval for training (open):** following the above, give the agent a fixed decision interval
  in seconds (action repeat / frame skip, e.g. `decision_dt = 0.1`) rather than one decision per env step,
  so that learned behaviour does not depend on `dt`.

## Known pitfalls

- `Arena.spawn_food` with `n_food = 0` raised (the attempt loop ran 0 times and fell through to the
  error); fixed with an early return, so food-free envs (`ZebrafishEnv(n_food=0)`) work.

- **Orbiting:** with any forward speed while turning hard, the turning radius (v / omega) can equal
  the distance to a nearby pellet, so the agent circles it with the pellet stuck outside the mouth
  cone and starves. Scripted policies must pivot in place (move = 0) when a nearby pellet is outside
  the cone (see `scripted_action` in `animate_env.py`). Expect RL agents to hit this early.
- `food_positions` is a numpy array: remove pellets with `Arena.remove_food(indices)` (all at once),
  never `del` or one-at-a-time removal.
- Use numpy (not `math`) in anything that may receive arrays; `wrap_angle` was switched to numpy for this.
- `ZebrafishEnv.can_eat` uses a 1e-9 tolerance so float sums of `dt` do not add an extra cooldown step.
- `WaypointFish` (in `animate_movement.py`): its `noise` parameter's effective std is
  `noise / sqrt(1 - smoothing^2)` (~0.34 for the defaults), not `noise`.
