"""
Evaluation of foraging policies: per-episode homeostasis metrics, and a comparison of policies on
the same seeded episodes. Shared by the training logs (train_ppo.HomeostasisCallback), the final
comparison table and viz/animate_ppo.py, so a metric means the same thing everywhere.

Every metric is a time average, an integral (sum * dt) or a physical count, so it does not change
with dt (see docs/PORTING_NOTES.md, "Evaluation metrics").
"""

import numpy as np

EVAL_EPISODES = 30
EVAL_SEED = 10_000 # evaluation episodes use seeds the training envs never see; equivalent to a "train-test" split
# per-episode totals (counts, seconds) that the ratio metrics are built from, kept so ratios can be
# pooled over episodes (sum of numerators / sum of denominators); left out of the printed table
RAW_KEYS = ("eat_opportunities", "eat_held_time", "eat_out_of_reach_time", "eat_during_cooldown_time")


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
    Ratios with nothing to divide by (an episode without EAT presses) are nan, not 0, so they are
    skipped by summarize instead of reading as "ate badly" or "aimed perfectly".
    Note: metrics with a raw step count are NOT dt-independent, be wary
    """
    glucose_trace = np.array([i["glucose"] for i in infos])
    actions = np.array([i["action"] for i in infos])     # (steps, 3)
    attempts = np.array([i["eat_attempted"] for i in infos])
    ready = np.array([i["eat_ready"] for i in infos])
    reach = np.array([i["food_in_reach"] for i in infos])
    info = infos[-1] # taking info of last step
    n_attempts = attempts.sum() # the total number of steps where the agent attempted to eat

    # EAT bouts (runs of consecutive steps with EAT held; a one-step tap is a bout of duration dt):
    # a bout lasting D s counts ceil(D / eat_cooldown) eat opportunities (one at its start, then one
    # per cooldown). Each bout is scored as if the cooldown were ready at its start, so possible_eats
    # is an upper bound on total_eaten, not the eats actually achievable: a bout started while the
    # cooldown from an earlier eat is still running counts as a wasted opportunity, on purpose.
    # 1e-9 keeps a bout of exactly one cooldown from rounding up to 2.
    starts = np.flatnonzero(attempts & ~np.r_[False, attempts[:-1]])     # first step of each bout
    ends = np.flatnonzero(attempts & ~np.r_[attempts[1:], False])        # last step of each bout
    bout_durations = (ends - starts + 1) * dt                    # s
    possible_eats = np.sum(np.ceil(bout_durations / eat_cooldown - 1e-9))

    # time after starvation counts at error = target, and every episode is averaged over episode_duration
    # units are error * s
    # multiplying by dt allows for dt-independence, because we scale the total error by how many timesteps were taken
    error_alive = np.sum(np.abs(glucose_trace - glucose_target)) * dt
    # error_dead adds max possible error (glucose_target away) after starvation, so the mean error is over the full episode duration, not just the alive time
    # this allows for fair comparison between starved vs. non-starved episodes
    error_dead = max(0.0, episode_duration - info["time"]) * glucose_target

    # EAT-held time and why presses fail, in s (reach first: no food at the mouth, then food at the
    # mouth but still cooling down); the rest (~eats * dt) are successful presses
    eat_held_time = n_attempts * dt
    out_of_reach_time = np.sum(attempts & ~reach) * dt
    during_cooldown_time = np.sum(attempts & reach & ~ready) * dt

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
        "eat_efficiency": float(info["total_eaten"] / possible_eats) if possible_eats else np.nan,
        # fractions of EAT-held time
        "eat_out_of_reach": float(out_of_reach_time / eat_held_time) if n_attempts else np.nan,
        "eat_during_cooldown": float(during_cooldown_time / eat_held_time) if n_attempts else np.nan,
        # raw totals (RAW_KEYS)
        "eat_opportunities": float(possible_eats),
        "eat_held_time": float(eat_held_time),
        "eat_out_of_reach_time": float(out_of_reach_time),
        "eat_during_cooldown_time": float(during_cooldown_time),
    }


def evaluate(policy, env_fn, n_episodes=EVAL_EPISODES, seed=EVAL_SEED):
    """
    Per-episode episode_metrics of policy(env, obs) -> action over n_episodes of env_fn(), e.g.
    training.train_ppo.make_env for the baselines, or make_ppo_env for PPO (actions in [-1, 1]).
    Episode i is reset with seed + i, so every policy faces the same arenas and starting glucose.

    Returns {metric: array of shape (n_episodes,)}, plus "seed" (the reset seed of each episode), so
    episodes can be probed individually or replayed (viz/animate_ppo.py --seed). Use summarize for
    the per-policy means.
    """
    # builds env to run policy in
    env = env_fn()
    base = env.unwrapped
    results = []
    for i in range(n_episodes): # evaluation episodes
        obs, info = env.reset(seed=seed + i) # reset env
        trace, done = [], False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(env, obs)) # do policy step
            trace.append(info) # record info
            done = terminated or truncated # check if episode done
        results.append(episode_metrics(trace, base.dt, base.glucose_target, base.max_steps * base.dt,
                                       base.eat_cooldown)) # record info for the entire episode
    per_episode = stack_episodes(results) # records every episode's info in a dict, with metrics as keys
    per_episode["seed"] = seed + np.arange(n_episodes) # stores the seed of each episode
    return per_episode


def stack_episodes(results):
    """
    List of per-episode dicts (e.g. episode_metrics outputs) -> {key: array of shape (n_episodes,)}.
    """
    return {key: np.array([r[key] for r in results]) for key in results[0]}


def summarize(per_episode, skip=("seed", "timestep", "env")):
    """
    One number per metric from {metric: array (n_episodes,)} (evaluate's output, or a stack_episodes
    of the training episodes), as {metric: float}. Labels in skip are dropped.
    Mean over the episodes where the metric is defined (nan = undefined, e.g. no EAT presses);
    nan if it is defined in none. Only for the printed table and the training logs, which need one
    number per metric; analysis (pooled ratios, conditioning, distributions) works on the saved
    per-episode arrays instead.
    """
    summary = {}
    for key, values in per_episode.items():
        if key in skip:
            continue
        defined = values[~np.isnan(values)]
        summary[key] = float(np.mean(defined)) if defined.size else np.nan
    return summary


def ppo_policy(model):
    """
    Deterministic (mean) PPO action, in [-1, 1]; evaluate it on make_ppo_env.
    
    PPO policies return a Gaussian over actions; during training, actions are sampled from it.
    But during evaluation, we just want the mean of the Gaussian (deterministic=True), which represents
    the policy's best guess of what to do in that state.
    """
    return lambda env, obs: model.predict(obs, deterministic=True)[0]


def print_comparison(rows):
    """
    Table of {policy name: metrics dict}, one row per policy. RAW_KEYS totals are left out.
    """
    keys = [k for k in next(iter(rows.values())) if k not in RAW_KEYS]
    widths = [max(len(k), 8) + 2 for k in keys]
    print(f"\n{'policy':<10}" + "".join(f"{k:>{w}}" for k, w in zip(keys, widths)))
    for name, metrics in rows.items():
        print(f"{name:<10}" + "".join(f"{metrics[k]:>{w}.3f}" for k, w in zip(keys, widths)))
