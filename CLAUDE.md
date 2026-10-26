# embodied_zfo

Continuous-space, continuous-time homeostatic foraging environment for a zebrafish-like agent.
It ports the discrete-time `GridWorld` from `../embodied_organs` to the continuous setting of
`../wef` (`wef/onpolicy/custom/fish/`), keeping embodied_organs' structure and dropping wef's
multi-agent, electric-sensing and curriculum code.

**Read `docs/PORTING_NOTES.md` at the start of a session**: it holds the current status,
the design decisions (with reasons), open questions and the next step.

## Layout

| File | Role |
|---|---|
| `movement.py` | Pure functions: `wrap_angle`, `propose_pose` (first-order kinematics), `enforce_bounds`, `step_pose` |
| `feeding.py` | Pure functions: `food_in_reach`, `nearest_food_in_reach` (eat radius + mouth cone) |
| `metabolism.py` | Pure functions: glucose dynamics (`basal_decay`, `movement_cost`, `update_glucose`), `drive`, `drive_reward`; placeholder rate constants at the top |
| `sensing.py` | Pure functions: `egocentric_bearing`, `nearest_food_observation` (k nearest pellets as proximity, cos/sin bearing, in_reach) |
| `arena.py` | `Arena`: walls, food as an `(N, 2)` array, `spawn_food`, `remove_food`, `draw` |
| `env.py` | `ZebrafishEnv(gymnasium.Env)`: combines the modules above; step order is eat -> move -> glucose -> reward -> end |
| `train_ppo.py` | SB3 PPO training (`make_ppo_env` = RescaleAction to [-1, 1]), `episode_metrics`, and the random / scripted / PPO evaluation; writes `runs/NAME/` (gitignored) |
| `test_*.py` | pytest suites, one per module |
| `plot_*.py`, `animate_*.py`, `compare_steering.py` | Visual checks; write to `figures/` (gitignored). `animate_ppo.py` replays a trained model on an evaluation episode |

## Conventions

- **Pure modules + thin env.** Physics, feeding and metabolism are pure functions that take state as
  arguments and return new values (never mutate inputs). `env.py` only orchestrates them;
  `Arena` owns world state. Keep new logic in this shape.
- **Time is physical.** All rates are per second and `dt` is explicit; never bake `dt` into
  constants (wef's `cfg.py` does; we deliberately don't). Durations such as the eating cooldown and
  episode length are in seconds. A change should not alter the dynamics when `dt` changes, beyond
  sampling; check this when adding anything time-dependent.
- **Angles:** radians, 0 = +x, counterclockwise positive, wrapped to [-pi, pi] with `wrap_angle`
  (numpy-based, works on arrays). Observations should be egocentric (relative to heading).
- **Placeholders are labeled.** Metabolic rates in `metabolism.py` and `EAT_COOLDOWN` in `env.py` are
  placeholders until set from the zebrafish literature; keep them named and documented.
- **Style:** match the existing code (docstrings on every function, short inline comments,
  descriptive names, numpy vectorization over loops).

## Environment and commands

- Conda env `embodied_zfo`, defined by `environment.yml`. Run everything with
  `conda run -n embodied_zfo ...`.
- **Every package must be tracked in `environment.yml`.** Installing is fine, but do it by adding the
  package to `environment.yml` first (conda-forge, or under a `pip:` section if it is pip-only) and
  then running `conda env update -n embodied_zfo -f environment.yml --prune`. Never `conda install` or
  `pip install` directly, so the file stays the single record of dependencies. Mention new
  packages to the user.
- Tests: `conda run -n embodied_zfo python -m pytest -q` (all should pass before committing).
- Visual checks: `conda run -n embodied_zfo python animate_env.py` (and the other `plot_*` /
  `animate_*` scripts); outputs go to `figures/`.

## Keeping notes current

When a porting step is completed or a design decision is made, update `docs/PORTING_NOTES.md`
(status, decision + reason, open questions, next step) in the same session.
