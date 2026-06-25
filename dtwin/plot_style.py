"""
plot_style.py
=============
A single place that defines the publication look of every figure, so the whole
set is visually consistent (fonts, sizes, grid, colour cycle).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# colour-blind-friendly palette
COLORS = {
    "true": "#1a1a1a",
    "est": "#1D9E75",     # teal
    "band": "#9FE1CB",
    "nominal": "#888780",
    "accent": "#0C447C",  # navy
    "warn": "#D85A30",    # coral
    "amber": "#EF9F27",
    "purple": "#7F77DD",
}


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "legend.fontsize": 8.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "lines.linewidth": 1.6,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def finish(ax, title=None, xlabel=None, ylabel=None, legend=True):
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
    return ax
