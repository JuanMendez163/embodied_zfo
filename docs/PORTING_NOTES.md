# Porting notes

Status, decisions and next steps for porting embodied_organs' `GridWorld` to a continuous
environment. Update this file whenever a step is completed or a decision is made.

_Last updated: 2026-10-26_

## Status

| Step | State | Where |
|---|---|---|
| 1. Movement | Done | `zfo/movement.py`, `tests/test_movement.py`, `viz/plot_movement.py`, `viz/animate_movement.py`, `viz/compare_steering.py` |
| 2a. Food placement | Done | `zfo/arena.py`, `tests/test_arena.py`, `viz/plot_food.py` |
| 2b. Eating rule | Done | `zfo/feeding.py`, `tests/test_feeding.py`, `viz/animate_feeding.py` |
| 3a. Glucose dynamics (was 2c) | Done | `zfo/metabolism.py`, `tests/test_metabolism.py` |
| 3b.1 Gymnasium env dynamics | Done | `zfo/env.py`, `tests/test_env.py`, `viz/animate_env.py` |
| 3b.2 Observation, option A (nearest k) | Done | `zfo/sensing.py`, `tests/test_sensing.py`, `zfo/env.py` (`obs_mode="nearest"`) |
| 3c. Pre-training fixes, first PPO run | Done | `zfo/env.py`, `training/train_ppo.py`, `runs/ppo/` (gitignored) |
| 3d.1 Inspect the trained agent | Done | `viz/animate_ppo.py` |
| Repo reorganized into packages | Done | `zfo/`, `training/`, `viz/`, `tests/`, `pyproject.toml` |
| **3d.2 Make the task harder (food ceiling first)** | **Next** | `zfo/arena.py`, `zfo/env.py` |

80 tests pass (`pytest -q`). Paths are relative to the repo root; files were moved into packages on
2026-10-26 (layout and commands in `CLAUDE.md`). The reorganization is not committed yet.

## First PPO run (2026-10-26)

1M steps (123 updates, ~3 min at ~6300 steps/s), seed 0, defaults in `train_ppo.py`. Evaluation on
30 seeded episodes (deterministic PPO actions):

| policy | starved | target error | eaten | move | abs turn | EAT rate | EAT eff. | out of reach |
|---|---|---|---|---|---|---|---|---|
| random | 1.00 | 41.2 | 0.4 | 0.51 | 0.50 | 0.50 | 0.003 | 1.00 |
| scripted (sees env state) | 0.00 | 3.7 | 17.9 | 0.46 | 0.15 | 0.35 | 0.20 | 0.96 |
| PPO | 0.00 | 3.7 | 18.4 | 0.34 | 0.39 | 0.46 | 0.16 | 0.97 |

- Learns fast: no starvation in training after ~57k steps; training error 43 -> ~6 by ~500k, then flat.
  Training error (sampled actions) stays above evaluation (mean actions) because std stays ~0.8.
- Matches the scripted forager on homeostasis. 3.7 is probably near the floor: each pellet gives a
  +10 sawtooth (mean abs error ~2.5 even if perfect), plus the transient from glucose_start in [20, 80].
- **Food ceiling:** training episodes eat ~19.5 of 20 pellets, so food availability limits both policies.
- **EAT not precise:** holds EAT ~46% of the time, 97% of it out of reach; expected while EAT is free.
- Swims slower but turns more than the scripted forager (abs turn 0.39 vs 0.15); worth watching
  for the orbiting pitfall.
- Healthy PPO stats: explained variance ~0.95-0.97, clip fraction 0.05-0.13, approx KL ~0.006.
- Single seed and 30 episodes: not yet a robust comparison.
- **Behaviour** (`viz/animate_ppo.py`, eval seed 10000, start glucose 26): hunger-gated foraging. When
  glucose falls to ~45 it **pivots in place** (|turn| = 1, move = 0), then **dashes straight** (move ~1,
  turn ~0) with EAT held, eats, then **rests** (move -> 0) with a slow drift in turn while glucose
  decays. So the extra |turn| is pivots plus idle drift, not orbiting: it found the pivot-then-move
  strategy that `zfo.policies.forage_action` needed hand-coded. EAT is gated by hunger (held during approaches, off
  while resting), not by reach. Occasional double eats overshoot to ~65. Ate all 20 pellets; the arena
  was empty by ~111 s.

## Next step: 3d.2 make the task harder

1. **Remove the food ceiling** (first): respawn eaten pellets (wef has regrowth) and/or lengthen
   episodes, so `total_eaten` and the target error can separate policies again. Re-run
   `training/train_ppo.py` and compare with the first run.
2. Harder sensing: limited `sensing_range`, `n_nearest` > 1.
3. Several seeds per configuration before trusting small differences.
4. Optional: a small cost per EAT attempt, if precise eating (tapping) should emerge.

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
- **Training setup** (`training/train_ppo.py`, SB3 PPO, MlpPolicy defaults: separate 64x64 tanh actor/critic):
  - **gamma = exp(-dt / tau), tau = 30 s** (~0.9967 at dt = 0.1), so the planning horizon is in
    seconds; SB3's default 0.99 is a 10 s horizon at dt = 0.1, shorter than the glucose half-life.
  - **Actions rescaled to [-1, 1]** with `RescaleAction` in `make_ppo_env`, used for both training and
    evaluation (one place for the translation). PPO's Gaussian starts at mean 0, std 1; on the raw
    [0, 1] move/eat ranges half the samples would clip to move = 0. The env keeps physical units.
  - **`glucose_start=(20, 80)`** in training and evaluation so the agent also experiences satiety;
    the env default stays at the target (tests and animations rely on it).
  - **No VecNormalize:** observations are already in [-1, 1] and the reward is scaled in the env.
  - 8 envs x `n_steps` 1024 = 8192 transitions per update, minibatch 256, 10 epochs, lambda 0.95,
    CPU (small MLPs). Evaluation seeds start at 10000, which training never uses.
- **Package layout** (2026-10-26): `zfo/` (env, pure modules, scripted baselines in `policies.py`),
  `training/` (`train_ppo.py`, `metrics.py`), `viz/` (visual checks, shared `style.py`), `tests/`.
  Dependencies point one way (`training`, `viz` -> `zfo`), so the env can be used without SB3 or
  plotting code. Before this, `train_ppo.py` imported its scripted baseline from a video script.
  Editable install (`-e .` in `environment.yml`, packages listed in `pyproject.toml`) so imports work
  from any directory; output dirs (`figures/`, `runs/`) are anchored to the repo root.
- **Step info for metrics:** `info` reports the applied (clipped) `action`, plus `eat_ready` and
  `food_in_reach` recorded before eating (= the agent's observed can_eat / in_reach).

- **Evaluation metrics** (`training.metrics.episode_metrics`, shared by the training logs and the final
  random / scripted / PPO table on the same seeded episodes):
  - `glucose_target_error` = mean |g - target| over the **full episode duration**, counting time after
    starvation at error = target. A per-step mean over alive time made starving early look good.
  - `starved` (0/1, averages to the starvation rate), `time`, `total_eaten`.
  - Diagnostics, averaged over alive time: `glucose_mean`/`glucose_std` (read only with `starved`),
    `mean_move_cmd`, `mean_abs_turn_cmd`, `eat_attempt_rate`.
  - `eat_efficiency` = eats / sum over EAT bouts of ceil(bout duration / eat_cooldown), in [0, 1].
    Bout-based because the first eat of a bout needs no wait: the time-based version
    (eats * cooldown / EAT-held time) gave a precise tapper 5 at dt = 0.1 and 10 at dt = 0.05.
    Scripted forager ~0.19, precise tapper 1.0 at every dt.
  - Why EAT presses fail, as fractions of EAT-held time, reach first: `eat_out_of_reach` (no pellet in
    the mouth cone) and `eat_during_cooldown` (pellet in reach, cooldown running: spamming at the food).
    The rest (~eats * dt) are successful presses. Uses `info["food_in_reach"]` / `info["eat_ready"]`,
    recorded before eating (= the agent's in_reach / can_eat). Scripted: 0.96 / 0.00; tapper: 0 / 0.
    Spamming through the cooldown costs the agent nothing (failed EATs do not restart it), so do not
    expect `eat_during_cooldown` to fall unless EAT is given a cost.
  - Every metric is a time average, an integral (sum * dt), or a physical count; step counts are not
    dt-independent.

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
- ~~dt-independence check for the full env~~ done (2026-10-26): `forage_action` on the same 10 seeded
  episodes at dt = 0.1, 0.05, 0.02 gives the same metrics within ~1-3% (target error ~3.5, 18.3
  pellets, no starvation). Per-step counts are not dt-independent (EAT success per *step* scaled with
  dt), hence the bout-based `eat_efficiency`.
- **Training settings that are per step** (not yet tied to seconds): GAE `lambda` (0.95 ~ a 2 s blend
  at dt = 0.1), `n_steps` (1024 = 102 s per env per rollout), total timesteps, and per-step Gaussian
  exploration noise (averages out faster at smaller dt). Gamma is already set from seconds.
- **Food runs out:** the scripted forager eats ~18-19 of 20 pellets in 120 s and PPO up to all 20, so
  `total_eaten` hits a ceiling and food limits performance (next step).
- **EAT is free:** failed attempts cost nothing, so PPO holds EAT whenever hungry (~46% of the time).
  A small per-attempt cost, or restarting the cooldown on failed attempts, would reward precision.

- **k > 1 and a limited sensing range:** try after the first training run. With k = 1 the
  "nearest" pellet can switch between steps when two are equidistant (dithering).
  With a limited range, food out of sight may call for a recurrent policy (memory).

- SB3's `check_env` still warns that the env's action space is not symmetric [-1, 1]; expected,
  since the rescale lives in `make_ppo_env` (see Training setup).

## Known pitfalls

- **Orbiting:** with any forward speed while turning hard, the turning radius (v / omega) can equal
  the distance to a nearby pellet, so the agent circles it with the pellet stuck outside the mouth
  cone and starves. Scripted policies must pivot in place (move = 0) when a nearby pellet is outside
  the cone (see `forage_action` in `zfo/policies.py`). The first PPO agent learned to pivot in place
  by itself (see Behaviour), but watch for this when changing movement or sensing.
- `food_positions` is a numpy array: remove pellets with `Arena.remove_food(indices)` (all at once),
  never `del` or one-at-a-time removal.
- Use numpy (not `math`) in anything that may receive arrays; `wrap_angle` was switched to numpy for this.
- `ZebrafishEnv.can_eat` uses a 1e-9 tolerance so float sums of `dt` do not add an extra cooldown step.
- Gymnasium 1.x wrappers do not forward attributes: on a `RescaleAction`-wrapped env use
  `env.unwrapped.dt` etc. (`info` passes through unchanged).
- `WaypointFish` (in `animate_movement.py`): its `noise` parameter's effective std is
  `noise / sqrt(1 - smoothing^2)` (~0.34 for the defaults), not `noise`.
