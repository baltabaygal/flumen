"""Generate the publication-quality figure set from the frozen bundle."""

import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from flumen.model import load_model
from flumen.model.scale import loc_scale
from flumen.training.data import DATASETS, available_splits, contexts


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TOKENS = {"surface":"#FCFCFD", "panel":"#FFFFFF", "ink":"#1F2430",
          "muted":"#6F768A", "grid":"#E6E8F0", "axis":"#D7DBE7"}
BLUE = {"xlight":"#EAF1FE", "light":"#CEDFFE", "base":"#A3BEFA",
        "mid":"#5477C4", "dark":"#2E4780"}
ORANGE = {"xlight":"#FFEDDE", "light":"#FFBDA1", "base":"#F0986E",
          "mid":"#CC6F47", "dark":"#804126"}
NEUTRAL = {"light":"#E2E5EA", "base":"#C5CAD3", "mid":"#7A828F",
           "dark":"#464C55"}
FONT = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]


def theme():
    sns.set_theme(style="whitegrid", context="paper", rc={
        "figure.facecolor": TOKENS["surface"], "axes.facecolor": TOKENS["panel"],
        "axes.edgecolor": TOKENS["axis"], "axes.labelcolor": TOKENS["ink"],
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": TOKENS["grid"], "grid.linewidth": .8,
        "font.family": "sans-serif", "font.sans-serif": FONT,
        "xtick.color": TOKENS["muted"], "ytick.color": TOKENS["muted"]})


def header(fig, axes, title, subtitle):
    left = min(ax.get_position().x0 for ax in np.ravel(axes))
    fig.text(left, .975, textwrap.fill(title, 75), ha="left", va="top",
             fontsize=17, weight="semibold", color=TOKENS["ink"])
    fig.text(left, .925, textwrap.fill(subtitle, 125), ha="left", va="top",
             fontsize=10, color=TOKENS["muted"])
    # Small fixed research blossom, kept clear of titles and plot marks.
    center = np.array([.965, .958]); radius = .011
    for angle in np.linspace(0, 2*np.pi, 6)[:-1]:
        dot = plt.Circle(center + radius*np.array([np.cos(angle), np.sin(angle)]),
                         .0045, transform=fig.transFigure, facecolor=BLUE["base"],
                         edgecolor=BLUE["dark"], linewidth=.5, clip_on=False)
        fig.add_artist(dot)


def finish(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=220, facecolor=TOKENS["surface"])
    fig.savefig(OUT / f"{stem}.pdf", facecolor=TOKENS["surface"])
    plt.close(fig)


def band_of(z):
    edges = [0, .3, .7, 1.5, 4, np.inf]
    labels = [r"$<0.3$", r"$0.3$–$0.7$", r"$0.7$–$1.5$",
              r"$1.5$–$4$", r"$\geq4$"]
    return pd.cut(z, edges, right=False, labels=labels)


def generalization_figure():
    validation = pd.read_csv(ROOT / "evaluation/results/validation_per_context.csv")
    test = pd.read_csv(ROOT / "evaluation/results/test_per_context.csv")
    validation["split"] = "Validation"; test["split"] = "Test"
    frame = pd.concat([validation[["z","kl_y","split"]],
                       test[["z","kl_y","split"]]], ignore_index=True)
    frame["Redshift band"] = band_of(frame["z"])
    order = list(frame["Redshift band"].cat.categories)
    palette = {"Validation": BLUE["light"], "Test": ORANGE["base"]}

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), gridspec_kw={"width_ratios":[1.45,1]})
    fig.subplots_adjust(top=.79, bottom=.16, left=.08, right=.97, wspace=.34)
    sns.boxplot(data=frame, x="Redshift band", y="kl_y", hue="split", order=order,
                palette=palette, showfliers=False, linewidth=.9, ax=axes[0])
    axes[0].set_yscale("log"); axes[0].set_ylim(3e-4, 3.5e-2)
    axes[0].set_xlabel(r"Source redshift $z_s$"); axes[0].set_ylabel(r"Per-configuration $KL_y$")
    axes[0].legend(title="", frameon=False, loc="upper left", ncol=2)
    axes[0].text(.01,.02,"Boxes: median and interquartile range; whiskers: 1.5× IQR",
                 transform=axes[0].transAxes, fontsize=7.5, color=TOKENS["muted"])

    metrics = ["Median", "90th percentile", "99th percentile", "Maximum"]
    values = {}
    for split, part in frame.groupby("split"):
        values[split] = [part.kl_y.median(), part.kl_y.quantile(.9),
                         part.kl_y.quantile(.99), part.kl_y.max()]
    y = np.arange(len(metrics))[::-1]
    axes[1].hlines(y, values["Validation"], values["Test"], color=NEUTRAL["base"], lw=1)
    axes[1].scatter(values["Validation"], y, s=48, facecolor=BLUE["light"],
                    edgecolor=BLUE["dark"], label="Validation", zorder=3)
    axes[1].scatter(values["Test"], y, s=48, facecolor=ORANGE["base"],
                    edgecolor=ORANGE["dark"], label="Test", zorder=3)
    axes[1].set_xscale("log"); axes[1].set_xlim(1.4e-3, 3.4e-2)
    axes[1].set_yticks(y, metrics); axes[1].set_xlabel(r"$KL_y$")
    axes[1].set_ylabel("")
    for val, yy in zip(values["Test"], y):
        axes[1].annotate(f"{val:.4f}", (val, yy), xytext=(5,0),
                         textcoords="offset points", va="center", fontsize=7.5,
                         color=ORANGE["dark"])
    header(fig, axes, "Held-out test performance tracks validation",
           "Width-relative KL in fixed y bins; all 972 validation configurations and all 973 untouched checkerboard test configurations.")
    fig.text(.08,.035,"Source: frozen production bundle, evaluated 2026-09-03. Lower is better.",
             fontsize=8, color=TOKENS["muted"])
    finish(fig, "generalization_certificate")


def landscape_figure():
    frame = pd.read_csv(ROOT / "evaluation/results/test_per_context.csv")
    frame["S8"] = frame.sigma8*np.sqrt(frame.OmegaM/.3)
    frame["log10_kl"] = np.log10(frame.kl_y)
    cmap = LinearSegmentedColormap.from_list("blue_kl",
        [TOKENS["panel"], BLUE["xlight"], BLUE["light"], BLUE["mid"], BLUE["dark"]])
    vmin, vmax = np.quantile(frame.log10_kl, [.02,.99])
    fig, ax = plt.subplots(figsize=(10.2, 6.2)); fig.subplots_adjust(top=.79,bottom=.15,left=.1,right=.88)
    marks = ax.scatter(frame.z, frame.S8, c=frame.log10_kl, cmap=cmap,
                       norm=Normalize(vmin,vmax), s=29, edgecolor=BLUE["dark"],
                       linewidth=.25, alpha=.88)
    ax.set_xscale("log"); ax.set_xlabel(r"Source redshift $z_s$")
    ax.set_ylabel(r"Structure amplitude $S_8=\sigma_8\sqrt{\Omega_m/0.3}$")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    cb = fig.colorbar(marks, ax=ax, pad=.025); cb.set_label(r"$\log_{10}(KL_y)$")
    worst = frame.loc[frame.kl_y.idxmax()]
    ax.scatter([worst.z],[worst.S8],s=120,marker="*",facecolor=ORANGE["base"],
               edgecolor=ORANGE["dark"],linewidth=.8,zorder=5)
    ax.annotate(f"maximum KL = {worst.kl_y:.4f}\n$z_s$ = {worst.z:.2f}",
                (worst.z,worst.S8),xytext=(-92,-34),textcoords="offset points",
                arrowprops={"arrowstyle":"-","color":NEUTRAL["dark"],"lw":.8},
                fontsize=8,color=TOKENS["ink"],bbox={"facecolor":"white","edgecolor":TOKENS["axis"],"pad":4})
    header(fig, [ax], "Most test configurations remain in the low-KL regime",
           "All 973 untouched test contexts; each point is one cosmology/redshift configuration and color encodes width-relative KL.")
    fig.text(.1,.035,"The orange star marks the maximum; color is clipped at the 2nd and 99th percentiles for readability.",
             fontsize=8,color=TOKENS["muted"])
    finish(fig, "test_parameter_landscape")


def representative_atlas(show_data=True):
    dataset = available_splits(DATASETS[0])["test"]
    ctx = contexts(dataset)
    targets = np.array([.22,.5,1.,3.5,5.,10.])
    fid = np.array([.67,.3,.85,.0493,.965,3.402])
    scales = np.array([.25,.35,1.1,.04,.05,.2])
    chosen = []
    for target in targets:
        score = ((np.log(ctx[:,0])-np.log(target))/.08)**2 \
                + np.sum(((ctx[:,1:]-fid)/scales)**2,axis=1)
        chosen.append(int(np.argmin(score)))
    model = load_model()
    scale_fit = torch.load(ROOT/"model/checkpoints/boundary_body.pt",
                           map_location="cpu",weights_only=False)["stats"]["scale_fit"]
    test_metrics = pd.read_csv(ROOT/"evaluation/results/test_per_context.csv").set_index("configuration_index")
    fig, axes = plt.subplots(2,3,figsize=(13.2,7.7)); fig.subplots_adjust(top=.81,bottom=.1,left=.075,right=.98,hspace=.34,wspace=.27)
    for ax, index in zip(axes.flat, chosen):
        z, theta = float(ctx[index,0]), tuple(ctx[index,1:])
        rays = dataset["lnmu"][index,:int(dataset["valid_counts"][index])]
        m,s = (float(v) for v in loc_scale(z,theta,scale_fit))
        y_edges = np.linspace(-4.2,8.2,56); x_edges=m+s*y_edges; mu_edges=np.exp(x_edges)
        counts,_=np.histogram(rays,x_edges); centers=np.sqrt(mu_edges[:-1]*mu_edges[1:])
        density=counts/(len(rays)*np.diff(mu_edges)); reliable=counts>=3
        grid_y=np.linspace(-4.35,8.35,900); mu=np.exp(m+s*grid_y)
        ax.plot(mu,model.pdf_mu(mu,z,theta),color=BLUE["mid"],lw=1.65,label="Production model")
        if show_data:
            error=np.zeros_like(density); error[reliable]=density[reliable]/np.sqrt(counts[reliable])
            ax.errorbar(centers[reliable],density[reliable],yerr=error[reliable],fmt="o",
                        ms=4.3,mfc=NEUTRAL["dark"],mec=TOKENS["ink"],mew=.4,
                        ecolor=NEUTRAL["base"],elinewidth=.6,capsize=0,label="Test rays")
        ax.set_yscale("log"); ax.set_ylim(1e-5,None); ax.set_xlim(mu.min(),mu.max())
        kl=float(test_metrics.loc[index,"kl_y"])
        ax.text(.96,.92,rf"$z_s={z:.2f}$"+"\n"+rf"$KL_y={kl:.4f}$",transform=ax.transAxes,
                ha="right",va="top",fontsize=9,color=TOKENS["ink"])
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        span = mu.max()-mu.min()
        if span < .2:
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        elif span < 2:
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        else:
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    for ax in axes[:,0]: ax.set_ylabel(r"$p(\mu\mid z_s,\theta)$")
    for ax in axes[-1,:]: ax.set_xlabel(r"Magnification $\mu$")
    handles=[Line2D([0],[0],color=BLUE["mid"],lw=1.65,label="Production model")]
    if show_data: handles.append(Line2D([0],[0],marker="o",ls="",mfc=NEUTRAL["dark"],mec=TOKENS["ink"],ms=5,label="Held-out test rays"))
    fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.075,.88),frameon=False,ncol=2)
    title = r"Held-out PDFs remain accurate from $z_s=0.2$ to $10$"
    subtitle = "Nearest-to-fiducial untouched test configurations; simulation bins require at least three rays and error bars are Poisson. Each panel uses its local physical-magnification range."
    if not show_data:
        subtitle = "Frozen model at the same six nearest-to-fiducial test contexts; physical magnification on x and density per unit magnification on y."
    header(fig, axes, title, subtitle)
    fig.text(.075,.025,"Displayed range corresponds to −4.35 < y < 8.35; log-density floor is 10⁻⁵.",fontsize=8,color=TOKENS["muted"])
    finish(fig, "representative_pdfs_with_data" if show_data else "representative_pdfs_model_only")


def main():
    theme(); generalization_figure(); landscape_figure()
    representative_atlas(True); representative_atlas(False)
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
