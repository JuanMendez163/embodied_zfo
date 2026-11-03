# Porting notes

Status, decisions and next steps for porting embodied_organs' `GridWorld` to a continuous
environment. Update this file whenever a step is completed or a decision is made.

_Last updated: 2026-11-03_

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
| Per-episode metrics saved (training + evaluation) | Done | `training/metrics.py`, `training/train_ppo.py`, `tests/test_metrics.py` |
| Analysis notebooks started (2026-10-29) | Done | `notebooks/ppo_analysis.ipynb`; `viz/animate_ppo.py` split into `load_model` / `make_animation` / `save_animation` so notebooks reuse it; command panel now shows only EAT held |
| Policy probing (2026-10-29) | Done | `training/probe.py`, `tests/test_probe.py`, notebook section "Policy probing" |
| Learning curves (2026-10-30) | Done | `viz/plot_training.py`, run `ppo_logged` (same config as `ppo`, with `train_episodes.npz`). Episodes are grouped by the rollout they ended in (one policy version each). Starvation ends within ~6 of 123 rollouts (~49k env steps); after that survival is capped at 120 s, so later learning needs `glucose_target_error` or reward |
| Observation-flag ablation (2026-11-02) | Done | `ZebrafishEnv(observe_can_eat=, observe_in_reach=)`, `train_ppo.py --hide`, runs `hide_can_eat`, `hide_in_reach`, `hide_both`; `viz/compare_training.py` (runs overlaid: survival / starved / eaten with the first-100k zoom, plus a `_homeostasis` figure of target error / EAT attempt rate; efficiency dropped as noise-dominated); results below |
| Stricter EAT cost: every press restarts the cooldown (2026-11-03) | Done | `ZebrafishEnv(eat_press_resets_cooldown=True)`, `train_ppo.py --eat-press-resets-cooldown`, run `eat_press_cost`; results below |
| EAT costs with both flags hidden (2026-11-03) | Done | runs `nocost_hide_both` (retrain of `hide_both` with strike metrics; reproduces it exactly), `eat_cost_hide_both`, `eat_press_cost_hide_both`; results below |
| **3d.2 Make the task harder (food ceiling first)** | **Next** | `zfo/arena.py`, `zfo/env.py` |

109 tests pass (`pytest -q`). Paths are relative to the repo root; files were moved into packages on
2026-10-26 (layout and commands in `CLAUDE.md`). The reorganization is committed; the per-episode metrics work (2026-10-27) is not committed yet.

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

## Observation-flag ablation (2026-11-02)

Question: can PPO learn without being told the cooldown has passed (`can_eat`) or that a pellet is in
reach (`in_reach`)? `ZebrafishEnv(observe_can_eat=False, observe_in_reach=False)` drop those columns
(the obs shrinks; nothing is zeroed). `train_ppo.py --hide can_eat in_reach` sets them and writes
`runs/NAME/env_kwargs.json`, which `load_env_kwargs` reads back so `viz/animate_ppo.py` and
`training/probe.py` rebuild the same layout. Seed 0, 1M steps, otherwise the `ppo_logged` config
(the "full" baseline).

| run | eval starved (det.) | eval error (det.) | eval error (stoch.) | eaten (stoch.) | P(EAT) d=1.9 / d=2.1 |
|---|---|---|---|---|---|
| full (`ppo_logged`) | 0.00 | 3.7 | 7.0 | 20.0 | 0.90 / 0.77 |
| no can_eat | 0.00 | 6.0 | 5.8 | 19.6 | 0.72 / 0.50 |
| no in_reach | **1.00** | 37.6 | 7.3 | 20.0 | 0.94 / 0.94 |
| neither | 0.00 | 6.8 | 6.1 | 19.6 | 0.51 / 0.51 |

(30 eval episodes; "stoch." = sampled actions, as in training. P(EAT): pellet straight ahead at
distance d, glucose 45; eat_radius = 2.)

- **Learning is unaffected.** Training curves (error, survival) overlap for all four
  (`figures/compare_obs_flags.png`). First rollout without starvation, i.e. PPO updates needed: full 5,
  no can_eat 6, neither 6, no in_reach 7 (49k-65k steps). One or two updates apart on one seed, so
  not a real difference yet. Training error over the last 10 rollouts 5.5-6.5 in every run.
- **can_eat is redundant in this env:** pellets spawn >= 10 apart and top speed is 10/s, so the next
  pellet is >= 1 s away, longer than the 0.5 s cooldown. Revisit if the cooldown becomes longer than
  the travel time between pellets (or food is clustered).
- **in_reach is used when given, not learned when hidden.** With the flag, P(EAT) steps up at the reach
  boundary; without it, P(EAT) is flat in distance and the EAT attempt rate drifts up during training
  (~0.4 with the flag vs ~0.7 without, by 1M steps). The agent never extracts the reach boundary
  (proximity > 1 - 2/141 ~ 0.986, a thin sliver) from proximity and bearing, because a failed EAT is free:
  holding EAT on every approach costs nothing. Testing whether reach *can* be learned needs an EAT cost.
- **The deterministic starvation of `hide_in_reach` is an evaluation artifact, not a learning failure.**
  Its stochastic policy eats all 20 pellets. The mean action with food ahead is move +0.81, turn -0.85
  (turning radius ~3 units), i.e. it orbits (see Known pitfalls); turn noise (std ~0.65) breaks the orbit
  in training. Every run has some turn bias with food ahead (full -0.38, neither +0.57); this one is just
  large enough to trap the mean policy. Single seed: do not attribute it to hiding in_reach.

## EAT cost: failed attempts restart the cooldown (2026-11-02)

Run `eat_cost` (`--eat-attempt-resets-cooldown`, full observation, seed 0, 1M steps) vs `ppo_logged` (same
config without the cost). Figures: `viz/compare_training.py --names ppo_logged eat_cost --out eat_cost`.

| run | first rollout w/o starvation | train error (last 10 rollouts) | eval error mean / sampled | eval glucose mean (mean action) | attempt rate (train, last 10) | out of reach (train) |
|---|---|---|---|---|---|---|
| no cost (`ppo_logged`) | 5 (~49k steps) | 6.5 | 3.7 / 7.0 | 50.7 | 0.47 | 0.97 |
| EAT cost (`eat_cost`) | 8 (~74k steps) | 7.0 | 10.6 / 6.6 | 58.7 | 0.42 | 0.91 |

- Learning is slower early (~3 updates later to stop starving; survival flat at ~51 s for rollouts 1-4),
  final homeostasis about the same (sampled eval 6.6 vs 7.0). One seed.
- The mean-action policy overeats (glucose ~59, eats ~20); sampled actions do not. Another case of
  deterministic evaluation diverging from the trained (stochastic) policy.
- The EAT rule barely changed: P(hold EAT) ~0.9 in reach, ~0.78 beyond, not glucose-gated, in both runs
  (probe, pellet straight ahead). It still holds EAT on approaches rather than tapping. What changed is the
  approach speed: with the cost, move falls with glucose (0.91 at 35 -> 0.61 at 65, pellet 20 ahead);
  without, it rises (0.83 -> 0.96). Likely why: a held EAT strikes every 0.5 s, and a pellet straight ahead
  is in reach for only ~0.2 s at full speed, so a slower approach raises the hit rate.
- **Strike metrics** (added 2026-11-02, `training/metrics.py`: `eat_strike_rate` per s alive,
  `eat_strike_miss`, raw `eat_strikes`). A strike = a press with the cooldown passed, counted as if each
  strike restarted the cooldown (own clock + the env's `ready`), so it is dt-independent with or without the
  cost, and equals `attempts & ready` when the cost is on (tested). Both runs were retrained (same seed,
  CPU) to log them; every existing metric reproduced exactly.
- **The cost raises strikes:** train (last 10 rollouts) 1.79/s vs 1.65/s, miss 0.908 vs 0.900; eval with
  the mean action 1.44 vs 1.06/s (no-cost values recounted 2026-11-03 with eats as strikes; were 1.54, 0.893, 0.93). Scripted forager: 1.32 vs 0.76/s (failing keeps it hungry, so it holds
  EAT more: attempt rate 0.62 vs 0.35). Reasons: a miss only costs if a pellet comes into reach within the
  next cooldown, so strikes far from food are free in practice (weak gradient; the slower approach targets
  the one part that does cost); and one press per 0.5 s window is enough to strike, so the cost agent's flat
  P(EAT) (~0.42-0.54 in every situation) strikes in more windows than the no-cost agent's concentrated one
  (0.64 food ahead, 0.31 no food). Training strike rates sit near the 2/s ceiling (sampled presses at
  p ~ 0.4 fill 92% of windows); the mean-action ordering is the same.
- Next options: `--eat-attempt-resets-cooldown --hide in_reach`; the stronger rule (presses during the
  cooldown also restart it), which removes the hold-and-slow solution (**built and run 2026-11-03**, next section); a reward penalty per missed strike
  (charges misses everywhere, not only near food; size must stay well below one pellet's reward); or less
  EAT exploration noise.

## Stricter EAT cost: every press restarts the cooldown (2026-11-03)

`ZebrafishEnv(eat_press_resets_cooldown=True)` / `train_ppo.py --eat-press-resets-cooldown` (default off).
Every EAT press restarts the cooldown, including presses made during it (`can_eat` is read before the
restart, so a ready press can still eat). Holding EAT therefore strikes once, at the start of the hold,
and eating needs a release of >= eat_cooldown followed by a tap. Implies the attempt rule. Run
`eat_press_cost` (full observation, seed 0, 1M steps). Figures: `viz/compare_training.py --names
ppo_logged eat_cost eat_press_cost --labels "no cost" "attempt cost" "press cost" --out eat_press_cost`.

| run | starvation ends at | eval error (mean action) | eval glucose | eval attempt rate | eval strike rate (/s) | eval strike miss | train attempt rate (last 10) |
|---|---|---|---|---|---|---|---|
| no cost (`ppo_logged`) | ~49k steps | 3.7 | 50.7 | 0.46 | 1.06 | 0.86 | 0.47 |
| attempt cost (`eat_cost`) | ~74k | 10.6 | 58.7 | 0.25 | 1.44 | 0.88 | 0.42 |
| press cost (`eat_press_cost`) | ~131k | 6.9 | 55.3 | 0.26 | 1.86 | 0.91 | 0.29 |

- **Baselines:** the scripted forager holds EAT whenever hungry, so under this rule it strikes once per
  hungry spell (usually far from food) and never eats: starves in 30/30 eval episodes (0 pellets). Random
  also starves (0.17 pellets). **Decision (2026-11-03): the scripted forager is dropped from the training
  evaluation** (`train_ppo.py` now prints random / PPO only). Comparisons are between trained agents from
  here on; `forage_action` stays only as the `viz/animate_env.py` demo of the movement repertoire. Its
  numbers in earlier sections are historical. No tapping forager will be built.
- **Learning is ~2x slower to stop starving** (~131k vs ~74k steps, ~16 vs ~9 rollouts); pellets eaten stay
  near 0 for the first ~100k steps, then catch up. Final training homeostasis matches the other runs (no
  starvation, ~19.7 pellets). Expected: the initial policy presses on ~50% of steps, and a strike needs
  4 press-free steps first (0.5^4 ~ 6% at dt = 0.1), so its own exploration noise keeps restarting the cooldown.
- **It learned a can_eat-locked tap.** Probe (glucose 45, pellet straight ahead): P(EAT) with
  can_eat = 1 / 0 is 0.84 / 0.20 in reach and 0.66 / 0.11 at d = 20 (the other runs do not gate on can_eat:
  `ppo_logged` 0.90 / 0.94, `eat_cost` 0.94 / 0.79). So the mean-action policy presses as the flag turns on,
  releases while it is off, and strikes once per cooldown: 1.86 strikes/s against the 2/s ceiling. The
  observation flag supplies the timing the memoryless policy otherwise lacks. Hiding `can_eat` under this rule
  would need the agent to time 0.5 s without memory, which a feedforward policy cannot (it would need
  recurrence or a frame stack): a sharp test of whether `can_eat` is needed.
- **Precision did not improve:** strike miss is 0.91 and with no food sensed P(EAT | ready) is 0.93, so it
  taps whenever ready, not when in reach. Same reason as for the attempt cost: a miss only costs if a pellet
  comes into reach within the next 0.5 s. Eval glucose overshoots (55 vs target 50), as with the attempt cost.
- **Metrics:** strikes still equal `attempts & ready` (a ready step had no press for a cooldown; tested in
  `test_metrics.py`). `eat_opportunities` / `eat_efficiency` assume one chance per cooldown of holding, but
  a hold now has one chance in total, so under this rule they are loose upper bounds; use the strike metrics.
- **dt:** the rule is physical (a press restarts a clock in seconds), but per-step exploration noise makes
  random press-free 0.5 s windows rarer at smaller dt, so early learning may be slower at smaller dt.
- Next options: `--eat-press-resets-cooldown --hide can_eat` (timing without the
  flag); a per-miss reward penalty to target precision.

## EAT costs with can_eat and in_reach both hidden (2026-11-03)

Same three EAT rules, `--hide can_eat in_reach`, seed 0, 1M steps. `nocost_hide_both` is `hide_both`
retrained to log strike metrics (every earlier metric reproduced exactly). Figures:
`viz/compare_training.py --names nocost_hide_both eat_cost_hide_both eat_press_cost_hide_both --labels
"no cost" "attempt cost" "press cost" --out eat_cost_hide_both`. Eval below uses **sampled** actions
(30 episodes, `model.predict(deterministic=False)`, `set_random_seed(0)`); the mean action is misleading here
(see the press-cost bullet).

| run | flags | starvation ends at | error | eaten | attempt rate | strikes/s | strike miss |
|---|---|---|---|---|---|---|---|
| no cost (`ppo_logged`) | both | ~49k steps | 7.0 | 20.0 | 0.49 | 1.67 | 0.90 |
| attempt cost (`eat_cost`) | both | ~74k | 6.6 | 19.5 | 0.44 | 1.79 | 0.91 |
| press cost (`eat_press_cost`) | both | ~131k | 6.5 | 19.6 | 0.29 | 1.21 | 0.87 |
| no cost (`nocost_hide_both`) | neither | ~57k | 6.1 | 19.6 | 0.73 | 1.91 | 0.91 |
| attempt cost (`eat_cost_hide_both`) | neither | ~74k | 7.4 | 19.7 | 0.33 | 1.23 | 0.86 |
| press cost (`eat_press_cost_hide_both`) | neither | **~172k** | 6.2 | 18.7 | 0.56 | **0.34** | **0.54** |

- **Homeostasis is the same in every run** (sampled error 6.1-7.4, no starvation). The costs change how
  the agent eats, not how well it regulates glucose. One seed each.
- **Reach is still not learned** under either cost: with both flags hidden, P(EAT) is identical at d = 1.5,
  1.9, 2.1 and 3 (pellet ahead), in all three runs. The open question from the ablation (can the reach
  boundary be extracted from proximity and bearing?) is answered "no" for these costs at 1M steps.
- **Attempt cost without flags: hunger-gated instead.** P(EAT), pellet ahead, glucose 35 / 45 / 55:
  0.89 / 0.65 / 0.32 (no-food-sensed 0.10 / 0.02 / 0.00); the no-cost run is milder (0.63 / 0.51 / 0.38),
  and with flags `eat_cost` is not glucose-gated at all (~0.94 at every glucose). Attempt rate 0.33 vs 0.73
  without cost. Mean-action eval works for this run (error 4.2, best mean-action error of the cost runs).
- **Press cost without flags: a stochastic tapper, broken under the mean action.** Mean-action eval:
  0 pellets, glucose 12.7, error 37.8 (survives 120 s only because decay is slow). With P(EAT) > 0.5
  the mean eat_cmd is above threshold, so the mean action holds EAT forever and the press rule never lets it
  strike. Sampled actions supply the release gaps: without can_eat (and no memory) it cannot time a release,
  so noise does it. **Evaluate press-cost runs without can_eat with sampled actions only.**
- **Its low miss rate comes from bearing, through an inverted press rate.** P(EAT) by bearing (d = 10,
  glucose 45): 0 deg 0.65, 30 deg 0.85, 60-90 deg 0.93, 135 deg 0.87, 180 deg 0.45; the no-cost and attempt-cost runs
  are the reverse (highest ahead). Under the press rule a strike needs 4 press-free steps then a press, so
  the per-step strike hazard is h(p) = p (1 - p)^4 (max at p = 0.2): h = 2e-5 at p = 0.93 vs 8e-3 at p = 0.65.
  Pressing *more* while the pellet is off-axis keeps the cooldown pinned and suppresses strikes there,
  so strikes land mostly with the pellet ahead (miss 0.54). Distance does not enter (d = 1.5 and 10 give the
  same P), and it approaches slower than the no-cost run (move 0.65 vs 0.81 at d = 5). This is per-step
  noise doing timing work, so it is tied to dt = 0.1 (the number of press-free steps per cooldown).
- **Not converged:** in training, strike miss is still falling at 1M steps (~0.75 at 0.7M -> ~0.57), and the
  attempt rate rises from ~0.3 to ~0.5 after ~0.6M while strikes fall (the strategy above taking over).
- **Strike metric fix (2026-11-03), affects the no-cost rows only.** Under no cost a failed strike does not
  restart the env's cooldown, so a held (or re-pressed) EAT can eat within eat_cooldown of a counted strike;
  66-73% of no-cost eats landed on uncounted steps (30 sampled eval episodes), so eats were not a subset of
  strikes and the old miss was slightly optimistic. `episode_metrics` now counts every eating step as a strike
  and restarts the strike clock there (eats restart the env cooldown under every rule). Decision: the strike
  metrics are **cost-rule accounting** (what pressing would cost if each strike were charged a cooldown); for
  a no-cost run they are a counterfactual, comparable across rules, and its native precision measure is
  `eat_out_of_reach` (fraction of EAT-held time). Cost-run values are unchanged (every eat was already a
  strike; verified). `ppo_logged` and `nocost_hide_both` retrained (same seed): every non-strike metric
  reproduced exactly. Strike rate can now exceed 1 / eat_cooldown slightly (an eat right after a miss).
  Figures `compare_eat_cost*`, `compare_eat_press_cost*`, `compare_eat_cost_hide_both*` regenerated;
  `compare_obs_flags*` not (its `hide_*` runs predate strike metrics).
- Next options: train `eat_press_cost_hide_both` longer (e.g. 3M) to see where miss settles; give the
  agent memory (frame stack or recurrent PPO) to test whether it then learns release-and-tap timing without
  can_eat; several seeds for the hunger-gating and bearing-inversion findings.

## Next step: 3d.2 make the task harder

1. **Remove the food ceiling** (first): respawn eaten pellets (wef has regrowth) and/or lengthen
   episodes, so `total_eaten` and the target error can separate policies again. Re-run
   `training/train_ppo.py` and compare with the first run.
2. Harder sensing: limited `sensing_range`, `n_nearest` > 1.
3. Several seeds per configuration before trusting small differences. **Deferred (2026-11-02):** while
   exploring qualitatively, one seed per configuration is enough; do the following for quantitative claims.
   - The training seed (`SEED` in `train_ppo.py`: initial weights, training arenas via `make_vec_env`,
     action sampling) is varied; evaluation seeds (`EVAL_SEED`) stay fixed, so every agent sees the same
     30 arenas. Needs a `--seed` flag in `train_ppo.py`.
   - Unit of analysis = the seed (one agent), not the episode: pooling episodes across seeds is
     pseudo-replication. Reduce each seed to one number per metric chosen in advance (e.g. area under the
     starvation curve over the first 100k steps, stochastic eval error, P(EAT) step at the reach boundary),
     then compare conditions on those numbers (mean + interval over seeds, individual seeds shown).
     Curves: per-seed rollout means as thin lines, mean over seeds bold, band = SE over seeds.
   - How many: 3 suffices for large, per-agent qualitative effects; for small differences (e.g. 1-2 PPO
     updates) use >= 5 (a 3-vs-3 permutation test cannot reach two-sided p < 0.1) and a continuous metric
     rather than "updates to no starvation" (an integer).
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
    The denominator counts eat opportunities (a one-step tap is one), scoring each bout as if the
    cooldown were ready at its start. So it is an upper bound on eats, not the achievable count:
    a bout started during the cooldown from an earlier eat is a wasted opportunity. Decided to
    keep this, since the metric is meant to reward eating when eating is possible and to penalize
    presses spent on eats that cannot happen.
  - Why EAT presses fail, as fractions of EAT-held time, reach first: `eat_out_of_reach` (no pellet in
    the mouth cone) and `eat_during_cooldown` (pellet in reach, cooldown running: spamming at the food).
    The rest (~eats * dt) are successful presses. Uses `info["food_in_reach"]` / `info["eat_ready"]`,
    recorded before eating (= the agent's in_reach / can_eat). Scripted: 0.96 / 0.00; tapper: 0 / 0.
    Spamming through the cooldown costs the agent nothing (failed EATs do not restart it), so do not
    expect `eat_during_cooldown` to fall unless EAT is given a cost.
  - Every metric is a time average, an integral (sum * dt), or a physical count; step counts are not
    dt-independent.
  - **Per-episode results are kept** (2026-10-27): `evaluate` returns `{metric: array (n_episodes,)}`
    plus `seed`, and `summarize` takes the means for the table. `train_ppo.py` saves them to
    `runs/NAME/eval.npz` (keys `policy/metric`). Reason: means hide bimodal outcomes (starved vs not)
    and are misleading for ratio metrics, so later analysis (conditioning on survival, pooled ratios,
    finding a bad seed to replay with `animate_ppo.py --seed`) needs the raw episodes.
  - **Training episodes are kept too** (2026-10-27): `HomeostasisCallback` still logs one summary
    per rollout to `progress.csv`, and also saves every training episode to
    `runs/NAME/train_episodes.npz`, one entry per episode: `timestep` (total env steps when it
    ended), `env` (which parallel env) and its `episode_metrics`. Re-saved every rollout. Learning
    curves are built from it afterwards (bin by timestep, choose the statistic per metric).
  - **Undefined ratios are nan** (2026-10-27): `eat_efficiency`, `eat_out_of_reach` and
    `eat_during_cooldown` are nan for an episode without EAT presses (was 0.0, which read as "ate
    badly" / "aimed perfectly"); `summarize` averages over the episodes where a metric is defined.
  - **Raw totals** (`RAW_KEYS`, 2026-10-27): `eat_opportunities` (count), `eat_held_time`,
    `eat_out_of_reach_time`, `eat_during_cooldown_time` (s), so ratios can be pooled over episodes
    (sum of numerators / sum of denominators). Left out of the printed table.
  - **Summaries stay simple; analysis is separate** (2026-10-27): `summarize` only gives the
    per-episode means needed by the printed table and `progress.csv`. Pooling, conditioning on
    survival and distributions are done later on the saved `.npz` files (an analysis notebook);
    `POOLED_RATIOS` in `metrics.py` lists which totals pool each ratio.
  - `tests/test_metrics.py` checks the metrics on hand-built step infos (tap, one-cooldown bout,
    no presses, failure split, starved error, dt-independence of opportunities, summarize).

## Open questions / to do

- ~~Notebook outputs in git~~ decided (2026-10-29): the replay video is saved to `figures/` and not
  embedded, so a run notebook stays small (~0.2 MB with plots). `nbstripout` only if that grows.
- **What the first PPO agent learned (probes, run `ppo`, 2026-10-29):** EAT depends on food geometry,
  hardly on glucose: mean eat_cmd > EAT_THRESHOLD whenever food is ahead (0.6-0.8 far, 0.9-1.1 in reach,
  falling only slightly with glucose), < threshold when food is behind; so it holds EAT on every
  approach (explains eat_out_of_reach ~0.97), as a failed press is free. Unclipped move and turn means
  shift roughly linearly with glucose, by about the same amount in every food geometry (an additive bias:
  move up, turn rightward, ~-0.02 turn per glucose unit). **Corrected 2026-10-30:** most of that shift is
  invisible in behavior. (1) The agent spends ~90% of the time at glucose 44-59 (1st-99th percentiles
  35-65, over 20 eval episodes); the rest of the 0-100 sweep is extrapolation. (2) There, move is clipped
  to 1 (food ahead) or 0 (food to the side, behind or in reach), and turn to -1 (food behind or in reach),
  whatever the glucose. The glucose burn rate of clip(mean) is flat or *falls* with glucose, so the earlier
  reading "swims faster above target to burn excess" does not hold. What survives clipping is a turn bias
  when food is ahead or to the side (food ahead far: turn +0.13 at 45, -0.15 at 58). Open: does that bias
  matter? Test by ablation: evaluate with the glucose input clamped to the target.

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
  **Built (2026-11-02):** `ZebrafishEnv(eat_attempt_resets_cooldown=True)` / `train_ppo.py
  --eat-attempt-resets-cooldown` (default off, so earlier runs replay unchanged). Decision: an *attempt* is
  a press made once the cooldown has passed; it restarts the cooldown eaten or not. Presses during the
  cooldown cost nothing (the stronger option, where they also restart it, would make holding EAT never
  succeed and force release-then-tap; harder exploration since the initial policy presses ~50% of steps).
  So holding EAT strikes once per eat_cooldown, which keeps `eat_efficiency`'s one-opportunity-per-cooldown
  accounting exact. Baselines under the rule (30 eval episodes): scripted (holds EAT when hungry) still
  survives, error 3.74 -> 5.42, eaten 17.9 -> 17.4, efficiency 0.198 -> 0.111; random unchanged (starves).
  This also makes `can_eat` informative (the cooldown now follows the agent's own presses), so the
  hide_can_eat ablation is worth rerunning under it.
  It is also why hiding `in_reach` changes nothing (ablation, 2026-11-02): rerun that ablation with an
  EAT cost to see whether reachability is learned from proximity and bearing.
- **Learning-curve points mix two policy versions** (2026-11-02, kept as is): episodes (1200 steps) are
  longer than a rollout per env (1024), so each env ends 0-1 episodes per rollout (5-8 per rollout,
  mean 8 * 1024 / 1200 = 6.83, set by the envs' phase offsets from early starvations), and each episode
  spans policies k-1 and k. Blur <= 1 update and identical across runs, so fine for relative comparisons;
  not for exact "updates needed" counts (resolution ~ +-1 update). Options, at the next config change
  (rerun all conditions then): `n_steps = round(episode_duration / dt)` (whole episodes per rollout while
  none starve; batch_size must divide 9600, e.g. 320), and/or log the rollout each episode *started* in
  (`HomeostasisCallback`), which stays correct with early termination or longer episodes.
- **`eat_efficiency` on training episodes is noise-dominated** (2026-11-02): with sampled actions (eat
  std ~0.47) EAT flickers into 2-3 step bouts, each counting as one opportunity, so opportunities
  (~275-286 per episode) track the press flip rate, not held time (48-83 s): efficiency is ~0.07 in all
  four ablation runs, while deterministic eval separates them (0.156 full vs 0.075 neither). Read
  `eat_attempt_rate` for training curves; efficiency only for deterministic or low-noise policies.
- **Deterministic evaluation can mislead** (2026-11-02): a policy whose mean action orbits can look
  starved in `evaluate(ppo_policy(model), ...)` while its sampled actions forage fine. Report both
  deterministic and stochastic evaluation when comparing runs, and use several seeds.

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
