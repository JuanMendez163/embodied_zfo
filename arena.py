"""
Continuous 2D arena: the space the agent moves in.

Simplified port of wef's Arena (arena.py): a fixed-size rectangle with walls, no food patches
or randomized size yet. Food positions will be added here in the next porting step,
replacing GridWorld's grid of Cell values.

The arena owns its geometry (size, walls, drawing); the agent's kinematics stay as pure
functions in movement.py, which take arena.size as an argument.
"""

import matplotlib.pyplot as plt
import numpy as np

from movement import enforce_bounds


class Arena:
    """
    Rectangle spanning [0, width] x [0, height] in arena units (e.g. cm).
    """

    def __init__(self, size=(100.0, 100.0), seed=None):
        self.size = (float(size[0]), float(size[1]))
        self.np_random = np.random.default_rng(seed)

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    def reset(self, seed=None):
        """
        Reset to the starting state. Only re-seeds the RNG if a seed is explicitly provided,
        matching GridWorld.reset. Nothing else to reset until food is added.
        """
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

    def clip(self, position, body_radius=0.0):
        """
        Keep a body of radius body_radius inside the walls. Returns (clipped_position, hit_wall).
        """
        return enforce_bounds(position, self.size, body_radius)

    def random_position(self, margin=0.0):
        """
        Uniformly sample a position at least margin away from every wall (e.g. an agent's start).
        """
        return np.array([
            self.np_random.uniform(margin, self.width - margin),
            self.np_random.uniform(margin, self.height - margin),
        ])

    def draw(self, ax, edgecolor="#52514e"):
        """
        Draw the arena walls on a matplotlib axis and fit the view to them.
        """
        ax.add_patch(plt.Rectangle((0, 0), self.width, self.height, fill=False,
                                   edgecolor=edgecolor, linewidth=1.5))
        pad = 0.03 * max(self.size)
        ax.set_xlim(-pad, self.width + pad)
        ax.set_ylim(-pad, self.height + pad)
        ax.set_aspect("equal")
