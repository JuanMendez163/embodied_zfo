"""
zfo: continuous-space, continuous-time homeostatic foraging environment for a zebrafish-like agent.
Pure modules (movement, feeding, metabolism, sensing), the Arena that holds world state, the
Gymnasium env that combines them, and scripted baseline policies.
"""

from zfo.env import ZebrafishEnv

__all__ = ["ZebrafishEnv"]
