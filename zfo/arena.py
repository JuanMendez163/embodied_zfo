"""
Continuous 2D arena: the space the agent moves in, and the food in it.

Simplified port of wef's Arena (arena.py): a fixed-size rectangle with walls and uniformly
placed food pellets, with optional regrowth (food_per_regrowth pellets, default 1, every
food_regrowth_interval seconds, placed uniformly, up to max_food), without wef's food patches, drift,
decay or randomized size.
Food replaces GridWorld's grid of Cell values: instead of marking cells as FOOD, the arena
keeps an (N, 2) array of pellet positions.

The arena owns what is in the world (geometry, food, drawing). What the agent does to it
(moving, eating) stays in pure functions elsewhere, e.g. movement.py, which takes arena.size.

Intended use at the start of an episode (mirrors GridWorld.reset: agent first, then food):
    arena.reset(seed)
    agent_position = arena.random_position(margin=...)
    arena.spawn_food(avoid=agent_position)
and at every step: arena.regrow_food(dt, avoid=agent_position).
"""

import matplotlib.pyplot as plt
import numpy as np

from zfo.movement import enforce_bounds


class Arena:
    """
    Rectangle spanning [0, width] x [0, height] in arena units (e.g. cm), holding food pellets.
    """

    def __init__(self, size=(100.0, 100.0), n_food=20, food_wall_margin=2.0,
                 food_min_distance=10.0, food_regrowth_interval=None, food_per_regrowth=1, max_food=None,
                 seed=None):
        self.size = (float(size[0]), float(size[1]))
        self.n_food = n_food                        # pellets placed by spawn_food (GridWorld's n_food)
        self.food_wall_margin = food_wall_margin    # keep pellets at least this far from every wall
        self.food_min_distance = food_min_distance  # keep pellets at least this far from the avoided point (agent)
        self.food_regrowth_interval = food_regrowth_interval   # s between regrowth ticks (None: no regrowth)
        self.food_per_regrowth = food_per_regrowth  # pellets grown at each tick (a batch)
        self.max_food = n_food if max_food is None else max_food   # no pellet grows while the arena holds this many
        self.time_since_regrowth = 0.0              # s since the last regrowth tick (restarts at reset)
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
        self.time_since_regrowth = 0.0

    def spawn_food(self, avoid=None):
        """
        Place self.n_food pellets uniformly at random, replacing any existing food.
        Every pellet is at least food_wall_margin from each wall and, if avoid (an (x, y) point,
        usually the agent's start) is given, at least food_min_distance from it.
        Sets self.food_positions to an (n_food, 2) array; draws only from self.np_random.
        """
        self.food_positions = self.sample_food_positions(self.n_food, avoid)

    def sample_food_positions(self, n, avoid=None):
        """
        n pellet positions, uniform over the arena minus food_wall_margin at each wall and, if avoid is
        given, minus a disc of radius food_min_distance around it (rejection sampling). Returns an
        (n, 2) array; draws only from self.np_random, one candidate (x, y) at a time.
        """
        positions = np.zeros((0, 2))
        for _ in range(1000 * max(n, 1)):
            if len(positions) >= n:
                return positions
            candidate = self.random_position(margin=self.food_wall_margin)
            if avoid is not None and np.linalg.norm(candidate - avoid) < self.food_min_distance:
                continue
            positions = np.vstack([positions, candidate])
        if len(positions) >= n:
            return positions
        raise ValueError(f"could not place {n} pellets: food_min_distance={self.food_min_distance} "
                        f"leaves too little room in arena {self.size} with margin {self.food_wall_margin}")

    def regrow_food(self, dt, avoid=None):
        """
        Advance the regrowth clock by dt seconds: food_per_regrowth pellets grow at every
        food_regrowth_interval since the episode began (t = interval, 2 * interval, ...), placed like
        spawn_food's (uniform, away from the walls and from avoid, usually the agent). Never grows past
        max_food: a batch that does not fit grows only what fits, and the rest is not saved for later.
        Returns the number of pellets added. The clock subtracts the interval rather than restarting at 0,
        so ticks fall at the same times for any dt (a step longer than the interval can hold several). With food_regrowth_interval None it does nothing and draws nothing from
        self.np_random, so episodes without regrowth replay exactly as before it existed.
        """
        if self.food_regrowth_interval is None:
            return 0
        self.time_since_regrowth += dt # measures how long it has been since last regrowth
        ticks = 0
        while self.time_since_regrowth >= self.food_regrowth_interval - 1e-9:   # has at least one interval passed? if so, take interval off the clock and count a tick
            self.time_since_regrowth -= self.food_regrowth_interval
            ticks += 1
        # a batch that does not fit grows only what fits under the cap (partial batch)
        n_new = min(ticks * self.food_per_regrowth, max(self.max_food - self.num_food, 0))
        if n_new > 0:
            self.food_positions = np.vstack([self.food_positions, self.sample_food_positions(n_new, avoid)])
        return n_new

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
