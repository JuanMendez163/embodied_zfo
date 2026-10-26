"""
Evaluation of foraging policies: per-episode homeostasis metrics, and a comparison of policies on
the same seeded episodes. Shared by the training logs (train_ppo.HomeostasisCallback), the final
comparison table and viz/animate_ppo.py, so a metric means the same thing everywhere.

Every metric is a time average, an integral (sum * dt) or a physical count, so it does not change
with dt (see docs/PORTING_NOTES.md, "Evaluation metrics").
"""

import numpy as np

EVAL_EPISODES = 30
EVAL_SEED = 10_000                # evaluation episodes use seeds the training envs never see


def episode_metrics(infos, dt, glucose_target, episode_duration, eat_cooldown):
    """
    Summary of how well one episode kept homeostasis, as a dict of floats (all averaged over episodes).
    infos: the env's info dict from every step of the episode, in order. Each has "glucose", "action"
    (move_cmd, turn_cmd, eat_cmd as applied), "eat_attempted", "n_eaten", and "eat_ready" (cooldown
    passed) / "food_in_reach" (pellet in the mouth cone), both at the start of the step, i.e. what the
    agent saw as can_eat / in_reach when it chose the action; the last one also has
    "total_eaten", "end_reason" and "time" (s) for the whole episode. dt in s.
    episode_duration: the time limit (s), so time after starvation can be counted (as glucose 0) and
    every episode is averaged over the same duration.
    eat_cooldown: s between successful eats, which caps the eating rate at 1 / eat_cooldown.
    Note: metrics with a raw step count are NOT dt-independent, be wary
    """
    glucose_trace = np.array([i["glucose"] for i in infos])
    actions = np.array([i["action"] for i in infos])     # (steps, 3)
    attempts = np.array([i["eat_attempted"] for i in infos])
    ready = np.array([i["eat_ready"] for i in infos])
    reach = np.array([i["food_in_reach"] for i in infos])
    info = infos[-1]
    n_attempts = attempts.sum() # the total number of steps where the agent attempted to eat

    # EAT bouts (runs of consecutive steps with EAT held): a bout lasting D s allows at most
    # ceil(D / eat_cooldown) eats (one at its start, then one per cooldown), so possible_eats bounds
    # total_eaten. 1e-9 keeps a bout of exactly one cooldown from rounding up to 2.
    starts = np.flatnonzero(attempts & ~np.r_[False, attempts[:-1]])     # first step of each bout
    ends = np.flatnonzero(attempts & ~np.r_[attempts[1:], False])        # last step of each bout
    bout_durations = (ends - starts + 1) * dt                    # s
    possible_eats = np.sum(np.ceil(bout_durations / eat_cooldown - 1e-9))

    # time after starvation counts at error = target, and every episode is averaged over episode_duration
    # units are error * s
    error_alive = np.sum(np.abs(glucose_trace - glucose_target)) * dt
    error_dead = max(0.0, episode_duration - info["time"]) * glucose_target

    return {
        "starved": float(info["end_reason"] == "starved"),
        # mean and std are over alive time only, so misleading for starved episodes; read with "starved"
        "glucose_mean": float(np.mean(glucose_trace)),
        "glucose_std": float(np.std(glucose_trace)),
        "glucose_target_error": float((error_alive + error_dead) / episode_duration),
        "total_eaten": float(info["total_eaten"]),
        "time": float(info["time"]),
        "mean_move_cmd": float(np.mean(actions[:, 0])),
        "mean_abs_turn_cmd": float(np.mean(np.abs(actions[:, 1]))),
        "eat_attempt_rate": float(n_attempts / len(infos)),                # fraction of time EAT is held
        "eat_efficiency": float(info["total_eaten"] / possible_eats) if possible_eats else 0.0,
        # why EAT presses fail, as fractions of EAT-held time (reach first: no food at the mouth,
        # then food at the mouth but still cooling down); the rest (~eats * dt) are successful presses
        "eat_out_of_reach": float(np.sum(attempts & ~reach) / n_attempts) if n_attempts else 0.0,
        "eat_during_cooldown": float(np.sum(attempts & reach & ~ready) / n_attempts) if n_attempts else 0.0,
    }


def evaluate(policy, env_fn, n_episodes=EVAL_EPISODES, seed=EVAL_SEED):
    """
    Mean episode_metrics of policy(env, obs) -> action over n_episodes of env_fn(), e.g.
    training.train_ppo.make_env for the baselines, or make_ppo_env for PPO (actions in [-1, 1]).
    Episode i is reset with seed + i, so every policy faces the same arenas and starting glucose.
    """
    env = env_fn()
    base = env.unwrapped
    results = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        trace, done = [], False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(env, obs))
            trace.append(info)
            done = terminated or truncated
        results.append(episode_metrics(trace, base.dt, base.glucose_target, base.max_steps * base.dt,
                                       base.eat_cooldown))
    return {key: float(np.mean([r[key] for r in results])) for key in results[0]}


def ppo_policy(model):
    """
    Deterministic (mean) PPO action, in [-1, 1]; evaluate it on make_ppo_env.
    """
    return lambda env, obs: model.predict(obs, deterministic=True)[0]


def print_comparison(rows):
    """
    Table of {policy name: metrics dict}, one row per policy.
    """
    keys = list(next(iter(rows.values())))
    widths = [max(len(k), 8) + 2 for k in keys]
    print(f"\n{'policy':<10}" + "".join(f"{k:>{w}}" for k, w in zip(keys, widths)))
    for name, metrics in rows.items():
        print(f"{name:<10}" + "".join(f"{metrics[k]:>{w}.3f}" for k, w in zip(keys, widths)))
