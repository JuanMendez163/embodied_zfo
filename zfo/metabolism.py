"""
Glucose dynamics, homeostatic drive and reward.

These are pure functions: they take glucose values as arguments instead of
reading them off an environment object, so they do not depend on whether
space is a grid or continuous.

Glucose dynamics are written in continuous time with an explicit dt (s), replacing
GridWorld.step_metabolism's per-step constants:
 - basal decay:    dg/dt = -decay_rate * g, so g(t + dt) = g(t) * exp(-decay_rate * dt)
                   (GridWorld's glucose_retention per step corresponds to exp(-decay_rate * dt))
 - movement cost:  (move_cost_rate * move_cmd**p + turn_cost_rate * |turn_cmd|**p) per second,
                   with p = move_cost_exponent; both swimming forward and turning use body/tail muscles
 - intake:         a fixed amount per pellet eaten, capped at glucose_max
 - starvation:     glucose below starvation_threshold is snapped to 0 (decay alone never reaches 0)

PLACEHOLDER DEFAULTS: the rates below are chosen to give sensible episode timescales, not taken
from the zebrafish literature yet. Replace them once there are measured or extrapolated values.
"""

import math

GLUCOSE_TARGET = 50.0         # homeostatic set point (GridWorld: 50)
GLUCOSE_MAX = 100.0           # stomach capacity (GridWorld: 100)
GLUCOSE_HALF_LIFE = 30.0      # s; time for glucose to halve with no food and no swimming (placeholder)
MOVE_COST_RATE = 1.0          # glucose/s at full speed (move_cmd = 1); ~ the basal loss at the target (placeholder)
TURN_COST_RATE = 0.5          # glucose/s at full turn rate (|turn_cmd| = 1); half the forward cost (placeholder)
MOVE_COST_EXPONENT = 2.0      # cost grows with speed**2 and turn rate**2, so gentle swimming/steering is cheap (placeholder)
INTAKE_AMOUNT = 10.0          # glucose per pellet eaten (GridWorld: 10)
STARVATION_THRESHOLD = 1.0    # glucose below this counts as starved (GridWorld: 1.0)


def decay_rate_from_half_life(half_life):
    """
    Basal decay rate (1/s) such that glucose halves every half_life seconds without food or movement.
    """
    return math.log(2) / half_life


def basal_decay(glucose, decay_rate, dt):
    """
    Glucose remaining after dt seconds of exponential decay at decay_rate (1/s).
    Exact for any dt, so results do not depend on the step size.
    """
    return glucose * math.exp(-decay_rate * dt)


def movement_cost(move_cmd, turn_cmd, dt, move_cost_rate=MOVE_COST_RATE, turn_cost_rate=TURN_COST_RATE,
                  move_cost_exponent=MOVE_COST_EXPONENT):
    """
    Glucose spent over dt seconds swimming at forward command move_cmd in [0, 1] while turning at
    turn_cmd in [-1, 1]. Left and right turns cost the same, so only |turn_cmd| matters.
    """
    forward = move_cost_rate * (move_cmd ** move_cost_exponent)
    turning = turn_cost_rate * (abs(turn_cmd) ** move_cost_exponent)
    return (forward + turning) * dt


def update_glucose(glucose, move_cmd, turn_cmd, n_eaten, dt,
                   decay_rate=decay_rate_from_half_life(GLUCOSE_HALF_LIFE),
                   move_cost_rate=MOVE_COST_RATE, turn_cost_rate=TURN_COST_RATE,
                   move_cost_exponent=MOVE_COST_EXPONENT,
                   intake_amount=INTAKE_AMOUNT, glucose_max=GLUCOSE_MAX,
                   starvation_threshold=STARVATION_THRESHOLD):
    """
    Glucose after one step of length dt in which the agent swam at move_cmd, turned at turn_cmd
    and ate n_eaten pellets.
    Combines basal_decay, movement_cost and intake; never negative, never above glucose_max,
    and snapped to 0 below starvation_threshold.
    """
    glucose = basal_decay(glucose, decay_rate, dt)
    glucose -= movement_cost(move_cmd, turn_cmd, dt, move_cost_rate, turn_cost_rate, move_cost_exponent)
    glucose += n_eaten * intake_amount
    glucose = max(0.0, min(glucose, glucose_max)) # clamping to [0, glucose_max]
    if glucose < starvation_threshold:
        glucose = 0.0
    return glucose


def drive(glucose, glucose_target, n=2, m=1):
    """
    Distance from homeostasis (Gutkin paper).
    (|g - g*|^n)^(1/m)
    n,m have important impact on shape of reward, drive curve.
    """
    return ((abs(glucose - glucose_target)) ** n) ** (1 / m)


def drive_reward(glucose_before, glucose_after, glucose_target, n=2, m=1):
    """
    Reward is the reduction in drive over one step, as in the Gutkin paper.
    reward = cur_drive - new_drive.
     - positive when the step moved glucose toward the target
     - negative when it moved glucose away (including overeating past the target)
    """
    return drive(glucose_before, glucose_target, n, m) - drive(glucose_after, glucose_target, n, m)
