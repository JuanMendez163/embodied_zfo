"""
Homeostatic drive and reward.

These are pure functions: they take glucose values as arguments instead of
reading them off an environment object, so they do not depend on whether
space is a grid or continuous.
"""


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
