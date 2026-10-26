"""
Sanity checks for metabolism.py. Run with: pytest test_metabolism.py
"""

import math

from zfo.metabolism import (GLUCOSE_MAX, GLUCOSE_TARGET, INTAKE_AMOUNT, STARVATION_THRESHOLD, basal_decay,
                        decay_rate_from_half_life, drive, drive_reward, movement_cost, update_glucose)

K = decay_rate_from_half_life(30.0)


def simulate(glucose, seconds, dt, move_cmd=0.0, turn_cmd=0.0):
    for _ in range(round(seconds / dt)):
        glucose = update_glucose(glucose, move_cmd, turn_cmd, 0, dt, decay_rate=K)
    return glucose


# --- basal decay and movement cost ---

def test_half_life():
    g = 40.0
    for _ in range(300):              # 300 steps of 0.1 s = one half-life
        g = basal_decay(g, K, 0.1)
    assert math.isclose(g, 20.0)


def test_decay_independent_of_dt():
    assert math.isclose(basal_decay(40.0, K, 10.0),
                        basal_decay(basal_decay(40.0, K, 5.0), K, 5.0))


def test_movement_cost_shape():
    assert movement_cost(0.0, 0.0, 0.1) == 0.0
    assert math.isclose(movement_cost(1.0, 0.0, 1.0, move_cost_rate=2.0), 2.0)
    # quadratic: half speed costs a quarter
    assert math.isclose(movement_cost(0.5, 0.0, 1.0, move_cost_exponent=2.0),
                        movement_cost(1.0, 0.0, 1.0, move_cost_exponent=2.0) / 4)


def test_turning_costs():
    assert math.isclose(movement_cost(0.0, 1.0, 1.0, turn_cost_rate=0.5), 0.5)
    # quadratic in turn rate too
    assert math.isclose(movement_cost(0.0, 0.5, 1.0), movement_cost(0.0, 1.0, 1.0) / 4)


def test_left_and_right_turns_cost_the_same():
    assert movement_cost(0.0, 0.7, 1.0) == movement_cost(0.0, -0.7, 1.0)


def test_swimming_and_turning_costs_add():
    assert math.isclose(movement_cost(0.8, -0.6, 1.0),
                        movement_cost(0.8, 0.0, 1.0) + movement_cost(0.0, -0.6, 1.0))


def test_zero_turn_cost_rate_ignores_turning():
    assert movement_cost(0.5, 1.0, 1.0, turn_cost_rate=0.0) == movement_cost(0.5, 0.0, 1.0)


# --- update_glucose ---

def test_idle_is_basal_decay_only():
    assert math.isclose(update_glucose(40.0, 0.0, 0.0, 0, 0.1, decay_rate=K), basal_decay(40.0, K, 0.1))


def test_swimming_costs_more_than_idling():
    assert update_glucose(40.0, 1.0, 0.0, 0, 0.1) < update_glucose(40.0, 0.0, 0.0, 0, 0.1)


def test_spinning_in_place_costs_more_than_idling():
    assert update_glucose(40.0, 0.0, 1.0, 0, 0.1) < update_glucose(40.0, 0.0, 0.0, 0, 0.1)


def test_eating_adds_intake():
    idle = update_glucose(30.0, 0.0, 0.0, 0, 0.1)
    assert math.isclose(update_glucose(30.0, 0.0, 0.0, 1, 0.1), idle + INTAKE_AMOUNT)
    assert math.isclose(update_glucose(30.0, 0.0, 0.0, 2, 0.1), idle + 2 * INTAKE_AMOUNT)


def test_capped_at_glucose_max():
    assert update_glucose(GLUCOSE_MAX - 1.0, 0.0, 0.0, 1, 0.1) == GLUCOSE_MAX


def test_starvation_snaps_to_zero():
    assert update_glucose(STARVATION_THRESHOLD * 0.99, 0.0, 0.0, 0, 0.1) == 0.0


def test_never_negative():
    assert update_glucose(0.5, 1.0, 1.0, 0, 1.0, move_cost_rate=100.0) == 0.0


def test_eating_can_rescue_from_starvation():
    # eating in the same step as dropping below the threshold keeps the agent alive
    assert update_glucose(STARVATION_THRESHOLD * 0.99, 0.0, 0.0, 1, 0.1) > STARVATION_THRESHOLD


def test_dynamics_roughly_independent_of_dt():
    # same 20 s of swimming at half speed while turning, sampled at two step sizes
    coarse = simulate(GLUCOSE_TARGET, 20.0, 0.1, move_cmd=0.5, turn_cmd=0.3)
    fine = simulate(GLUCOSE_TARGET, 20.0, 0.05, move_cmd=0.5, turn_cmd=0.3)
    assert math.isclose(coarse, fine, rel_tol=1e-2)


# --- drive and reward ---

def test_drive_zero_at_target():
    assert drive(GLUCOSE_TARGET, GLUCOSE_TARGET) == 0


def test_reward_sign():
    assert drive_reward(30.0, 40.0, GLUCOSE_TARGET) > 0     # toward the target
    assert drive_reward(40.0, 30.0, GLUCOSE_TARGET) < 0     # away from it
    assert drive_reward(55.0, 65.0, GLUCOSE_TARGET) < 0     # overeating past it
