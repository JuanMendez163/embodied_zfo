"""
Continuous 2D arena: the space the agent moves in, and the food in it.

Simplified port of wef's Arena (arena.py): a fixed-size rectangle with walls and uniformly
placed food pellets, without wef's food patches, drift, regrowth or randomized size.
Food replaces GridWorld's grid of Cell values: instead of marking cells as FOOD, the arena
keeps an (N, 2) array of pellet positions.

The arena owns what is in the world (geometry, food, drawing). What the agent does to it
(moving, eating) stays in pure functions elsewhere, e.g. movement.py, which takes arena.size.

Intended use at the start of an episode (mirrors GridWorld.reset: agent first, then food):
    arena.reset(seed)
    agent_position = arena.random_position(margin=...)
    arena.spawn_food(avoid=agent_position)
"""

import matplotlib.pyplot as plt
import numpy as np

from movement import enforce_bounds


class Arena:
    """
    Rectangle spanning [0, width] x [0, height] in arena units (e.g. cm), holding food pellets.
    """

    def __init__(self, size=(100.0, 100.0), n_food=20, food_wall_margin=2.0,
                 food_min_distance=10.0, seed=None):
        self.size = (float(size[0]), float(size[1]))
        self.n_food = n_food                        # pellets placed by spawn_food (GridWorld's n_food)
        self.food_wall_margin = food_wall_margin    # keep pellets at least this far from every wall
        self.food_min_distance = food_min_distance  # keep pellets at least this far from the avoided point (agent start)
        self.np_random = np.random.default_rng(seed)
        self.food_positions = np.zeros((0, 2))

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    @property
    def num_food(self):
        return len(self.food_positions)

    def reset(self, seed=None):
        """
        Reset to an empty arena. Only re-seeds the RNG if a seed is explicitly provided,
        matching GridWorld.reset. Call spawn_food afterwards to place pellets.
        """
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.food_positions = np.zeros((0, 2))

    def spawn_food(self, avoid=None):
        """
        Place self.n_food pellets uniformly at random, replacing any existing food.
        Every pellet is at least food_wall_margin from each wall and, if avoid (an (x, y) point,
        usually the agent's start) is given, at least food_min_distance from it.
        Sets self.food_positions to an (n_food, 2) array; draws only from self.np_random.
        """
        max_attempts = 1000 * self.n_food
        self.food_positions = np.zeros((0, 2))
        for _ in range(max_attempts):
            if len(self.food_positions) >= self.n_food:
                return
            candidate = self.random_position(margin=self.food_wall_margin)
            if avoid is not None and np.linalg.norm(candidate - avoid) < self.food_min_distance:
                continue
            self.food_positions = np.vstack([self.food_positions, candidate])
        raise ValueError(f"could not place {self.n_food} pellets: food_min_distance={self.food_min_distance} "
                        f"leaves too little room in arena {self.size} with margin {self.food_wall_margin}")

    def remove_food(self, indices):
        """
        Remove the pellets at the given indices into food_positions (e.g. the ones just eaten).
        Indices refer to the array before removal; duplicates and an empty list are fine.
        """
        self.food_positions = np.delete(self.food_positions, np.asarray(indices, dtype=int), axis=0)

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

    def draw(self, ax, edgecolor="#52514e", food_color="#1baf7a"):
        """
        Draw the arena walls and the current food on a matplotlib axis, and fit the view to them.
        Returns the food artist; to animate eating, call food_artist.set_offsets(arena.food_positions)
        each frame.
        """
        ax.add_patch(plt.Rectangle((0, 0), self.width, self.height, fill=False,
                                   edgecolor=edgecolor, linewidth=1.5))
        food_artist = ax.scatter(self.food_positions[:, 0], self.food_positions[:, 1], s=20,
                                 color=food_color, edgecolor="none", zorder=2)
        pad = 0.03 * max(self.size)
        ax.set_xlim(-pad, self.width + pad)
        ax.set_ylim(-pad, self.height + pad)
        ax.set_aspect("equal")
        return food_artist
