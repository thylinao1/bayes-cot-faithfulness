"""Shared dark house style for project figures, matching the website palette.

The project site (``docs/site`` and the mirrored ``website/``) is dark with a
fixed accent palette. Every generated figure is embedded on that site, so the
figures should read as one set rather than as stock matplotlib output. These
helpers pin matplotlib to the exact CSS custom-property tokens from
``styles.css`` and add the soft accent glow the site uses on its own elements.

Colour semantics are shared with the site on purpose:

    orange  decorative / natural direct effect (NDE)
    cyan    faithful / natural indirect effect (NIE)
    green   total effect, and the "recovers truth" marker

Both ``02_generate_figure.py`` and ``04_generate_sensitivity_figure.py`` import
from here, so the look stays consistent from a single source of truth.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

# Exact CSS custom-property tokens from the website (styles.css :root).
PALETTE: dict[str, str] = {
    "bg": "#0b0f17",       # --bg-2, the figure-card background on the site
    "bg_deep": "#07090f",  # --bg
    "panel": "#0e131c",    # --bg-tint
    "border": "#1f2632",   # --border
    "edge": "#5a6478",
    "text": "#e6ebf2",     # --text
    "text_2": "#a4adbd",   # --text-2
    "text_3": "#6a7384",   # --text-3
    "muted": "#8a93a4",    # --muted
    "orange": "#ff8c42",   # --orange : NDE / decorative path
    "cyan": "#4dd0e1",     # --cyan   : NIE / faithful path
    "green": "#a5e887",    # --green  : total effect / recovery marker
    "violet": "#b88dff",   # --violet
}


def use_house_style() -> None:
    """Pin matplotlib rcParams to the dark website palette. Call once per figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["bg"],
            "axes.edgecolor": PALETTE["edge"],
            "axes.labelcolor": PALETTE["text"],
            "xtick.color": PALETTE["text_2"],
            "ytick.color": PALETTE["text_2"],
            "text.color": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "grid.color": PALETTE["border"],
            "font.family": ["SF Pro Display", "Inter", "system-ui", "sans-serif"],
            "font.size": 11,
        }
    )


def glow_line(
    ax: Axes,
    x,
    y,
    color: str,
    lw: float = 2.4,
    n_glow: int = 5,
    glow_alpha: float = 0.05,
    zorder: float = 3.0,
    **kwargs,
) -> Line2D:
    """Plot a line with a soft neon glow, echoing the accent glows on the site.

    Draws a few wider, low-alpha copies behind a crisp line. Extra keyword
    arguments (marker, label, ...) apply to the crisp line. Returns it.
    """
    for i in range(n_glow, 0, -1):
        ax.plot(
            x, y, color=color, lw=lw + i * 1.7, alpha=glow_alpha,
            solid_capstyle="round", zorder=zorder - 0.01,
        )
    (line,) = ax.plot(x, y, color=color, lw=lw, zorder=zorder, **kwargs)
    return line


def style_legend(ax: Axes, **kwargs) -> None:
    """Add a frameless legend with palette-coloured text (readable on the dark bg)."""
    legend = ax.legend(framealpha=0.0, **kwargs)
    for text in legend.get_texts():
        text.set_color(PALETTE["text_2"])
