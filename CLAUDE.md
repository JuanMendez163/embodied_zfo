# embodied_zfo

Continuous-space, continuous-time homeostatic foraging environment for a zebrafish-like agent.
It ports the discrete-time `GridWorld` from `../embodied_organs` to the continuous setting of
`../wef` (`wef/onpolicy/custom/fish/`), keeping embodied_organs' structure and dropping wef's
multi-agent, electric-sensing and curriculum code.

**Read `docs/PORTING_NOTES.md` at the start of a session**: it holds the current status,
the design decisions (with reasons), open questions and the next step.

## Layout

Four packages, with dependencies pointing one way: `training/` and `viz/` import from `zfo/`;
`zfo/` imports from neither. The repo is installed editable (`pyproject.toml`, via `environment.yml`),
so `import zfo` works from any directory and scripts can be run by path.

| Path | Role |
|---|---|
| `zfo/movement.py` | Pure functions: `wrap_angle`, `propose_pose` (first-order kinematics), `enforce_bounds`, `step_pose` |
| `zfo/feeding.py` | Pure functions: `food_in_reach`, `nearest_food_in_reach` (eat radius + mouth cone) |
| `zfo/metabolism.py` | Pure functions: glucose dynamics (`basal_decay`, `movement_cost`, `update_glucose`), `drive`, `drive_reward`; placeholder rate constants at the top |
| `zfo/sensing.py` | Pure functions: `egocentric_bearing`, `nearest_food_observation` (k nearest pellets as proximity, cos/sin bearing, in_reach) |
| `zfo/arena.py` | `Arena`: walls, food as an `(N, 2)` array, `spawn_food`, `remove_food`, `draw` |
| `zfo/env.py` | `ZebrafishEnv(gymnasium.Env)` (also `from zfo import ZebrafishEnv`): combines the modules above; step order is eat -> move -> glucose -> reward -> end |
| `zfo/policies.py` | Scripted baselines as `policy(env, obs)`: `forage_action` (reads env state), `random_policy` |
| `training/train_ppo.py` | SB3 PPO training: config, `make_env` / `make_ppo_env` (RescaleAction to [-1, 1]), `HomeostasisCallback`; writes `runs/NAME/` (gitignored) |
| `training/metrics.py` | `episode_metrics`, `evaluate` (per-episode metrics on the same seeded episodes for every policy), `stack_episodes`, `summarize` (means for the table and logs only; analysis uses the saved per-episode arrays), `POOLED_RATIOS`, `ppo_policy`, `print_comparison` |
| `training/probe.py` | Policy probing: `probe_observations` (hand-built observations, one feature swept), `policy_gaussian` (actor mean and std in physical action units, unclipped), `eat_probability` |
| `viz/style.py` | Shared palette, `style_trace`, `FIGURES_DIR` (repo-root `figures/`, gitignored) |
| `viz/plot_*.py`, `viz/animate_*.py`, `viz/compare_steering.py` | Visual checks. `animate_ppo.py` replays a trained model on an evaluation episode; `plot_training.py --name NAME` draws learning curves from `train_episodes.npz` (per-rollout mean ± standard error, x = env steps over all envs) |
| `notebooks/*.ipynb` | Analysis notebooks (`ppo_analysis.ipynb`: evaluation episodes, replay video saved to `figures/` (not embedded), policy probing). Thin: they import from `training/` and `viz/`, so reusable code goes in those packages, not in cells |
| `tests/test_*.py` | pytest suites, one per `zfo` module, plus `test_metrics.py`, `test_train_ppo.py` and `test_probe.py` for `training/` |

## Conventions

- **Pure modules + thin env.** Physics, feeding and metabolism are pure functions that take state as
  arguments and return new values (never mutate inputs). `zfo/env.py` only orchestrates them;
  `Arena` owns world state. Keep new logic in this shape.
- **Time is physical.** All rates are per second and `dt` is explicit; never bake `dt` into
  constants (wef's `cfg.py` does; we deliberately don't). Durations such as the eating cooldown and
  episode length are in seconds. A change should not alter the dynamics when `dt` changes, beyond
  sampling; check this when adding anything time-dependent. This applies to evaluation metrics too:
  use time averages, integrals (sum * dt) or physical counts, never raw step counts; and to training
  settings (gamma is set from a horizon in seconds).
- **Angles:** radians, 0 = +x, counterclockwise positive, wrapped to [-pi, pi] with `wrap_angle`
  (numpy-based, works on arrays). Observations should be egocentric (relative to heading).
- **Placeholders are labeled.** Metabolic rates in `zfo/metabolism.py` and `EAT_COOLDOWN` in `zfo/env.py` are
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
- Visual checks: `conda run -n embodied_zfo python viz/animate_env.py` (and the other `viz/` scripts);
  outputs go to the repo-root `figures/` from any working directory.
- Training: `conda run -n embodied_zfo python training/train_ppo.py [--timesteps N] [--name NAME]` (~3 min for
  the default 1M steps; writes `runs/NAME/model.zip`, `progress.csv`, `train_episodes.npz` (every training episode's metrics, with timestep and env) and `eval.npz` (per-episode evaluation metrics), then prints the random /
  scripted / PPO table). SB3 prints a table every update, so redirect stdout to a log file.
  Replay a trained agent: `conda run -n embodied_zfo python viz/animate_ppo.py --name NAME [--seed S]`.
- Notebooks: open `notebooks/*.ipynb` with the `embodied_zfo` kernel (VS Code, or
  `conda run -n embodied_zfo jupyter lab`). To check one runs headless: `conda run -n embodied_zfo jupyter
  nbconvert --to notebook --execute NB.ipynb --output-dir <scratch dir>` (keeps outputs out of the repo copy).
- New modules go in the package they belong to; keep `zfo/` free of training and plotting code.
  If `pyproject.toml` gains a package, re-run the `conda env update` above.

## Keeping notes current

When a porting step is completed or a design decision is made, update `docs/PORTING_NOTES.md`
(status, decision + reason, open questions, next step) in the same session.

Claude edits only the docs it maintains: `docs/PORTING_NOTES.md` and this `CLAUDE.md`. Other files in
`docs/` (e.g. `docs/PPO + Env Notes.md`) are the user's own notes: read them, but never modify them;
point out anything outdated in the reply instead.
