"""Render the frozen production model over the 5 x 8 acceptance panel.

The bundled simulator reference is fixed: 400k image-plane rays per panel on a
common physical-magnification grid.  The historical box-high row is replaced by
the midpoint of the full six-dimensional training box, using its own packaged
ordinary-branch rays.
"""

import argparse
import os
from pathlib import Path
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gw-wl-production-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gw-wl-production-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from flumen.model import load_model


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "evaluation" / "reference" / "full_parameter_panel_reference.npz"
BOX_MIDPOINT = ROOT / "evaluation" / "reference" / "box_midpoint_rays_ordinary_branch.npz"
OUT = ROOT / "evaluation" / "figures"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
BLUE = {
    "xlight": "#EAF1FE",
    "light": "#CEDFFE",
    "base": "#A3BEFA",
    "mid": "#5477C4",
    "dark": "#2E4780",
}
NEUTRAL = {"base": "#C5CAD3", "mid": "#7A828F", "dark": "#464C55"}
FONT = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]


def use_theme():
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.72,
            "font.family": "sans-serif",
            "font.sans-serif": FONT,
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
        },
    )


def add_header(fig, axes, *, show_data, ymin):
    left = axes[0, 0].get_position().x0
    title = "Production emulator across the full parameter-space panel"
    subtitle = (
        "Five reference cosmologies × eight source redshifts; physical density "
        rf"$p(\mu)$ over $0.3\leq\mu\leq150$, shown above $10^{{{int(np.log10(ymin))}}}$. "
        + (
            "Points are fixed 400k-ray simulator references. The Box midpoint row "
            rf"retains only the ordinary branch $1-\kappa-|\gamma|>0$."
            if show_data
            else "Curves show the frozen production model without simulator overlays."
        )
    )
    fig.text(
        left,
        0.984,
        title,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.956,
        textwrap.fill(subtitle, 175),
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
        linespacing=1.18,
    )

    # Compact research blossom; deliberately outside the analytical field.
    center = np.array([0.975, 0.968])
    radius = 0.0085
    for angle in np.linspace(0, 2 * np.pi, 6)[:-1]:
        petal = plt.Circle(
            center + radius * np.array([np.cos(angle), np.sin(angle)]),
            0.0035,
            transform=fig.transFigure,
            facecolor=BLUE["base"],
            edgecolor=BLUE["dark"],
            linewidth=0.45,
            clip_on=False,
        )
        fig.add_artist(petal)


def load_references():
    ref = np.load(REFERENCE)
    counts = np.asarray(ref["counts"], dtype=float).copy()
    sizes = np.asarray(ref["sizes"], dtype=float).copy()
    edges = np.asarray(ref["edges"], dtype=float)
    contexts = np.asarray(ref["cosmos"], dtype=float)
    redshifts = np.asarray(ref["zs"], dtype=float)
    names = [str(value) for value in ref["names"]]

    midpoint = np.load(BOX_MIDPOINT)
    box_row = names.index("box hi-struct")
    theta = np.asarray(midpoint["theta"], dtype=float)
    contexts[box_row] = np.array(
        [theta[0], theta[1], theta[2], theta[3], theta[4], theta[5] / 1000.0]
    )
    names[box_row] = "Box midpoint"
    for k, z in enumerate(np.asarray(midpoint["zs"], dtype=float)):
        col = int(np.flatnonzero(np.isclose(redshifts, z))[0])
        rays = np.asarray(
            midpoint["rays"][midpoint["offsets"][k] : midpoint["offsets"][k + 1]],
            dtype=float,
        )
        counts[box_row * len(redshifts) + col], _ = np.histogram(np.exp(rays), bins=edges)
        sizes[box_row * len(redshifts) + col] = len(rays)

    return edges, contexts, redshifts, names, counts, sizes


def render(*, show_data, ymin):
    use_theme()
    OUT.mkdir(parents=True, exist_ok=True)
    edges, contexts, redshifts, names, counts, sizes = load_references()
    nrow, ncol = len(names), len(redshifts)
    model = load_model()
    mu = np.geomspace(0.3, 150.0, 1500)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = np.diff(edges)

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(27.5, 16.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        top=0.885,
        bottom=0.072,
        left=0.065,
        right=0.992,
        hspace=0.14,
        wspace=0.075,
    )

    for row, (name, theta) in enumerate(zip(names, contexts)):
        for col, z in enumerate(redshifts):
            ax = axes[row, col]
            density = np.asarray(model.pdf_mu(mu, float(z), tuple(theta)), dtype=float)
            sns.lineplot(
                x=mu,
                y=density,
                ax=ax,
                color=BLUE["mid"],
                linewidth=1.45,
                legend=False,
                sort=False,
                zorder=3,
            )

            if show_data:
                index = row * ncol + col
                bin_counts = counts[index]
                sim_density = bin_counts / (sizes[index] * widths)
                reliable = bin_counts >= 3
                weak = (bin_counts > 0) & (bin_counts < 3)
                sns.scatterplot(
                    x=centers[reliable],
                    y=sim_density[reliable],
                    ax=ax,
                    s=27,
                    color=NEUTRAL["dark"],
                    edgecolor=TOKENS["ink"],
                    linewidth=0.35,
                    alpha=0.92,
                    legend=False,
                    zorder=5,
                )
                sns.scatterplot(
                    x=centers[weak],
                    y=sim_density[weak],
                    ax=ax,
                    s=24,
                    facecolor=TOKENS["panel"],
                    edgecolor=NEUTRAL["dark"],
                    linewidth=0.65,
                    legend=False,
                    zorder=5,
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.3, 150.0)
            ax.set_ylim(ymin, 3e2)
            ax.tick_params(labelsize=7.5, length=2.5)
            ax.grid(True, which="major")
            ax.grid(False, which="minor")
            if row == 0:
                ax.set_title(rf"$z_s={z:g}$", fontsize=10.5, pad=6, color=TOKENS["ink"])
            if col == 0:
                ax.set_ylabel(name + "\n" + r"$p(\mu)$", fontsize=9, labelpad=8)
            else:
                ax.set_ylabel("")
            if row == nrow - 1:
                ax.set_xlabel(r"Magnification $\mu$", fontsize=8.5)
            else:
                ax.set_xlabel("")

    handles = [Line2D([0], [0], color=BLUE["mid"], lw=1.45, label="Production model")]
    if show_data:
        handles.extend(
            [
                Line2D(
                    [0], [0], marker="o", linestyle="", markersize=5.2,
                    markerfacecolor=NEUTRAL["dark"], markeredgecolor=TOKENS["ink"],
                    markeredgewidth=0.35, label="Simulator (≥3 rays)"
                ),
                Line2D(
                    [0], [0], marker="o", linestyle="", markersize=5.0,
                    markerfacecolor=TOKENS["panel"], markeredgecolor=NEUTRAL["dark"],
                    markeredgewidth=0.65, label="Simulator (1–2 rays)"
                ),
            ]
        )
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(axes[0, 0].get_position().x0, 0.927),
        frameon=False,
        ncol=len(handles),
        borderaxespad=0,
        fontsize=8.5,
        handlelength=2.3,
        columnspacing=1.5,
    )
    add_header(fig, axes, show_data=show_data, ymin=ymin)
    fig.text(
        axes[-1, 0].get_position().x0,
        0.018,
        "Source: frozen production bundle and fixed acceptance-panel simulations. Filled points require at least three rays per bin.",
        ha="left",
        fontsize=8,
        color=TOKENS["muted"],
    )

    ymin_tag = f"1e{int(np.log10(ymin))}"
    variant = "with_data" if show_data else "model_only"
    stem = OUT / f"full_parameter_atlas_mu_ymin_{ymin_tag}_{variant}"
    fig.savefig(stem.with_suffix(".png"), dpi=220, facecolor=TOKENS["surface"])
    fig.savefig(stem.with_suffix(".pdf"), facecolor=TOKENS["surface"])
    fig.savefig(stem.with_suffix(".svg"), facecolor=TOKENS["surface"])
    plt.close(fig)
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ymin", type=float, default=1e-3)
    args = parser.parse_args()
    if not 0 < args.ymin < 3e2:
        parser.error("--ymin must be positive and below 300")
    for show_data in (True, False):
        print(render(show_data=show_data, ymin=args.ymin))


if __name__ == "__main__":
    main()
