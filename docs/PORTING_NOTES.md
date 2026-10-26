# Porting notes

Status, decisions and next steps for porting embodied_organs' `GridWorld` to a continuous
environment. Update this file whenever a step is completed or a decision is made.

_Last updated: 2026-10-26_

## Status

| Step | State | Where |
|---|---|---|
| 1. Movement | Done | `movement.py`, `test_movement.py`, `plot_movement.py`, `animate_movement.py`, `compare_steering.py` |
| 2a. Food placement | Done | `arena.py`, `test_arena.py`, `plot_food.py` |
| 2b. Eating rule | Done | `feeding.py`, `test_feeding.py`, `animate_feeding.py` |
| 3a. Glucose dynamics (was 2c) | Done | `metabolism.py`, `test_metabolism.py` |
| 3b.1 Gymnasium env dynamics | Done | `env.py`, `test_env.py`, `animate_env.py` |
| 3b.2 Observation, option A (nearest k) | Done | `sensing.py`, `test_sensing.py`, `env.py` (`obs_mode="nearest"`) |
| **3c. Pre-training fixes, then PPO** | **Next** | `env.py`, `train_ppo.py` (to write) |

77 tests pass (`pytest -q`).

## Next step: 3c. pre-training fixes, then PPO

Before the first training run (see "Open questions" for why):
1. ~~Reward scale~~ done (`reward_scale`, see decisions).
2. **Discount from a horizon in seconds:** `gamma = exp(-dt / tau)`, tau ~ 30 s (gamma ~ 0.997 at
   dt = 0.1). The default 0.99 is a 10 s horizon at dt = 0.1, shorter than the glucose half-life.
3. ~~Randomize `glucose_start`~~ done: `glucose_start=(low, high)` samples uniformly at each reset.
   Default stays at the target (tests and animations rely on it); `train_ppo.py` should pass e.g. (20, 80).
4. **Add `stable-baselines3` and `pytorch` to `environment.yml`**, write `train_ppo.py`, and compare
   against a random policy (starves in ~64 s on average) and `scripted_action` in `animate_env.py`
   (metrics: time near target, pellets eaten, starvation rate).

Sector sensors (option B, wef style) remain an alternative `obs_mode`: S angular sectors, each
reporting `1 - d / range` of the nearest pellet in range (`EFishAgent._update_sensors` in
`wef/onpolicy/custom/fish/MAEFish.py`).

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
  Decay and movement cost are applied sequentially (operator splitting), not via the exact joint
  solution `g e^{-k dt} - (c/k)(1 - e^{-k dt})`; kept for simplicity since the error is ~k*dt/2
  (~0.1% per step at the current half-life and dt = 0.1 s). Revisit if dt grows or the half-life shrinks.
- **Reward** = reduction in drive (Gutkin / GridWorld); no separate starvation penalty.
  `terminated` = starved, `truncated` = time limit (episode length in seconds).
- **Observation (`obs_mode="nearest"`, default)** = `[glucose / glucose_max, can_eat]` followed by,
  for each of the `n_nearest` (k, default 1) nearest pellets within `sensing_range`,
  `[proximity, cos(bearing), sin(bearing), in_reach]`. With k = 1 this is the continuous port of
  GridWorld's Q-agent state (nearest-food distance + direction + glucose). Reasons:
  - **proximity `1 - d / range`**, not raw distance: inputs of order 1 for the network, and "nothing
    sensed" is naturally 0 (raw d has no natural value for absence). Equivalent to `d / range` up to a
    flip; proximity matches wef's sector sensors.
  - **No "food present" flag:** unsensed slots are all zeros ("nothing detected"), which is what a
    real sensor reports. The agent cannot know whether food exists beyond what it senses.
  - **Egocentric bearing as cos/sin:** continuous across +-pi (behind), and maps directly onto turn
    actions (GridWorld's allocentric N/S/E/W has no heading to refer to).
  - **`in_reach`** reuses `feeding.food_in_reach`, so it is 1 exactly when EAT could succeed
    (GridWorld's `O`); easier than learning the radius/cone threshold from dist and bearing.
  - **`sensing_range` defaults to the arena diagonal** (sees everything, closest to GridWorld) so a
    failed first training run is not due to limited sensing. A shorter range is more biological.
  - `obs_mode="internal"` (glucose, can_eat only) is kept as a blind baseline.
- **Reward scale:** reward = drive reduction / `reward_scale`, default = one pellet's worth
  (`drive(target - intake, target)` = 100 at the current defaults): eating from 40 to 50 gives +1,
  near starvation (5 -> 15) +8, starving totals about -25. A positive constant keeps every ratio
  between rewards (the drive's convexity, `drive_n`, is what makes hungry pellets worth more) and the
  optimal policy. Avoid reward *clipping* (SB3 `VecNormalize` clips at 10 by default), which would
  flatten near-starvation rewards.
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
- **dt-independence check for the full env:** run the same scripted forager at `dt = 0.1` and `0.05`
  and compare glucose traces over the same simulated time.

- **k > 1 and a limited sensing range:** try after the first training run. With k = 1 the
  "nearest" pellet can switch between steps when two are equidistant (dithering).
  With a limited range, food out of sight may call for a recurrent policy (memory).

## Known pitfalls

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
