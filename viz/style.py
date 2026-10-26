"""
Shared look for the env videos (animate_env.py, animate_ppo.py): palette, time-trace axes, and the
figures/ directory every visual check writes to.
"""

from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"   # repo-root figures/, wherever run from

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
AGENT = "#2a78d6"
FOOD = "#1baf7a"
TURN = "#eb6834"
EAT_MARK = "#008300"


def style_trace(ax, duration):
    """
    Time-trace axes (0 to duration s): surface background, light horizontal grid, muted ticks and spines.
    """
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, duration)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)
