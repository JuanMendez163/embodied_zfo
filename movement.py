"""
Agent kinematics in a continuous 2D arena.

Ported from wef's movement.py (first-order dynamics only), with the multi-agent
collision check, size/curriculum multipliers and eating freeze removed.

These are pure functions, like metabolism.py: they take the pose as arguments and
return a new pose instead of mutating an agent object.

Conventions:
 - position is an np.array([x, y]) in arena units (e.g. cm); the arena spans
   [0, width] x [0, height] ; says where the agent is in the arena
 - heading is in radians, 0 points along +x, positive turns counterclockwise,
   always wrapped to [-pi, pi] ; says which way the agent is facing
   -- wrapping to [-pi, pi] ensures a consistent representation of the same heading
 - commands are dimensionless: move_cmd in [0, 1] (forward only), turn_cmd in [-1, 1]
 - max_speed and max_turn_rate are physical rates (units/s, rad/s); dt (s) is kept
   explicit rather than pre-multiplied into the rates as wef's cfg.py does
"""

import math

import numpy as np


def wrap_angle(theta):
    """
    Wrap an angle to [-pi, pi] (stable for any input, unlike modulo arithmetic).
    """
    return math.atan2(math.sin(theta), math.cos(theta))


def propose_pose(position, heading, move_cmd, turn_cmd, max_speed, max_turn_rate, dt):
    """
    First-order dynamics: commands act directly as velocities for one step of length dt.
    Returns (proposed_position, proposed_heading). Walls are not checked here.
    """
    speed = move_cmd * max_speed # forward speed (units/s) ; move_cmd in [0, 1]
    turn_rate = turn_cmd * max_turn_rate # angular speed (rad/s) ; turn_cmd in [-1, 1]
    delta_heading = turn_rate * dt # move that rate per time step
    new_heading = wrap_angle(heading + delta_heading)
    # speed * dt is magnitude of forward movement ; cos + sin relate that to the heading
    proposed_position = position + np.array([math.cos(new_heading), math.sin(new_heading)]) * speed * dt

    return proposed_position, new_heading


def enforce_bounds(position, arena_size, body_radius=0.0):
    """
    Clip position so the agent's body stays inside the arena.
    Replaces GridWorld.in_bounds: instead of rejecting the move, the agent slides along the wall.
    Returns (clipped_position, hit_wall).
    """
    width, height = arena_size
    clipped = np.array([
        np.clip(position[0], body_radius, width - body_radius),
        np.clip(position[1], body_radius, height - body_radius),
    ], dtype=float)
    hit_wall = not np.array_equal(clipped, position)
    return clipped, hit_wall


def step_pose(position, heading, move_cmd, turn_cmd, max_speed, max_turn_rate, dt,
              arena_size, body_radius=0.0):
    """
    Advance the pose by one step: propose a move from the commands, then enforce walls.
    Heading is always updated, even when the wall blocks the position (agent can turn in place).
    Returns (new_position, new_heading, hit_wall).
    """
    proposed_position, new_heading = propose_pose(
        position, heading, move_cmd, turn_cmd, max_speed, max_turn_rate, dt)
    new_position, hit_wall = enforce_bounds(proposed_position, arena_size, body_radius)
    return new_position, new_heading, hit_wall
