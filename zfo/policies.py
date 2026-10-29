"""
Scripted baseline policies for ZebrafishEnv, as policy(env, obs) -> action in physical units
(move_cmd, turn_cmd, eat_cmd). Used as references for learned agents (training/train_ppo.py) and in
the demo videos (viz/animate_env.py).
"""

import math

import numpy as np

from zfo.movement import wrap_angle


def forage_action(env):
    """
    Hunger-gated foraging: steer to the nearest pellet and EAT while below the glucose target,
    otherwise cruise slowly and ignore food. Reads env state directly (all food positions), so it is a
    reference, not a fair competitor, once sensing is limited.

    This is just a sample policy to compare against, not trained or tuned
    """
    hungry = env.glucose < env.glucose_target
    food = env.arena.food_positions
    if not hungry or len(food) == 0:
        return [0.3, 0.1, 0.0]                      # sated: slow cruise, ignore food
    offsets = food - env.position
    distances = np.linalg.norm(offsets, axis=1)
    target = offsets[np.argmin(distances)]
    heading_error = wrap_angle(math.atan2(target[1], target[0]) - env.heading)
    turn_cmd = float(np.clip(heading_error / (math.pi / 2), -1.0, 1.0))
    move_cmd = 1.0 - abs(turn_cmd)
    # pivot in place when a nearby pellet is outside the mouth cone: any forward speed there makes the
    # agent orbit the pellet (turning radius ~ distance to it), keeping it beside the mouth forever
    if distances.min() < 2 * env.eat_radius and abs(heading_error) > env.mouth_angle / 2:
        move_cmd = 0.0
    return [move_cmd, turn_cmd, 1.0]                # hungry: try to eat whenever possible


def random_policy(seed=0):
    """
    Uniformly random commands, a lower bound.
    """
    rng = np.random.default_rng(seed)
    return lambda env, obs: rng.uniform(env.action_space.low, env.action_space.high)
