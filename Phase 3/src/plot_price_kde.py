#!/usr/bin/env python3
"""Phase 3 — price-distribution *density curves* (KDE) on the merged dataset.

  1) one chart, all rows, one smooth curve per STATUS (overlaid)
        -> figs/phase3_price_kde_by_status.png
  2) small multiples: one curve per DISTRICT, its own panel
        -> figs/phase3_price_kde_by_district.png

Density is estimated on log10(price) (prices span ~1-300B), drawn on a log price axis.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "phase3_merged_dataset.parquet"
FIGS = pathlib.Path(__file__).resolve().parents[1] / "figs"

BLUE, GREEN, RED, ORANGE, TEAL, PURPLE, GREY = (
    "#2a78d6", "#0a9a3c", "#c8342f", "#e08214", "#17a0a8", "#7b52c8", "#8a8f98")
INK, MUTED, GRID, SURF = "#1d2027", "#5b6675", "#e6e8ec", "#ffffff"
STATUS_COLOR = {"SOLD": GREEN, "REVOKED": ORANGE, "AVAILABLE": BLUE,
                "ON_STOCK": TEAL, "LOCKED": PURPLE, "UNKNOWN": GREY}
XT = [1, 2, 5, 10, 20, 50, 100, 200]
GRID_X = np.linspace(np.log10(0.7), np.log10(300), 400)   # log10 grid
PX = 10 ** GRID_X


def kde_curve(prices):
    lp = np.log10(prices[(prices > 0)].values)
    k = gaussian_kde(lp)
    return k(GRID_X)


def logx(ax):
    ax.set_xscale("log")
    ax.set_xlim(0.9, 300)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(length=0, colors=MUTED, labelsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def main():
    FIGS.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA)
    df = df[df["price_bil"] > 0].copy()
    df["status"] = df["status"].replace({"ON_STOCK": "AVAILABLE"})   # merge live inventory into AVAILABLE

    # ---------- 1) overlaid curves by STATUS ----------
    order = df.groupby("status")["price_bil"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    logx(ax)
    ax.set_xticks(XT)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    for s in order:
        g = df.loc[df.status == s, "price_bil"]
        y = kde_curve(g)
        c = STATUS_COLOR.get(s, GREY)
        ax.fill_between(PX, y, color=c, alpha=0.08, zorder=2)
        ax.plot(PX, y, color=c, lw=2.4, zorder=3, label=f"{s}  (n={len(g):,}, med {np.median(g):.1f}B)")
        ax.axvline(np.median(g), color=c, lw=1, ls=(0, (2, 2)), alpha=0.5, zorder=1)
    ax.set_yticks([])
    ax.set_ylabel("relative frequency (density)", color=MUTED, fontsize=10)
    ax.set_xlabel("Asking price  (billion ₫, log scale)", color=MUTED, fontsize=10)
    ax.set_title("Hà Nội asking-price distribution by listing status", color=INK,
                 fontsize=15, fontweight="bold", loc="left", pad=40)
    ax.text(0, 1.035, "Phase 3 merged set (n=27,800) · density on log10(price) · dashed line = median",
            transform=ax.transAxes, color=MUTED, fontsize=10.5, va="bottom")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=INK, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(FIGS / "phase3_price_kde_by_status.png", bbox_inches="tight", facecolor=SURF)
    plt.close(fig)

    # ---------- 2) small multiples by DISTRICT ----------
    vc = df["district_name"].value_counts()
    keep = vc[vc >= 100].index
    dd = df[df.district_name.isin(keep)].copy()
    order = dd.groupby("district_name")["price_bil"].median().sort_values(ascending=False).index.tolist()
    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.5 * nrow), dpi=150, sharex=True)
    axes = axes.ravel()
    ymax = 0
    curves = {}
    for d in order:
        g = dd.loc[dd.district_name == d, "price_bil"]
        curves[d] = (kde_curve(g), np.median(g), len(g))
        ymax = max(ymax, curves[d][0].max())
    for i, d in enumerate(order):
        ax = axes[i]
        y, med, n = curves[d]
        logx(ax)
        ax.set_xticks([1, 10, 100])
        ax.fill_between(PX, y, color=BLUE, alpha=0.16, zorder=2)
        ax.plot(PX, y, color=BLUE, lw=2, zorder=3)
        ax.axvline(med, color=INK, lw=1.4, ls=(0, (2, 2)), zorder=4)
        ax.set_ylim(0, ymax * 1.08)
        ax.set_yticks([])
        name = d.replace("Q. ", "").replace("H. ", "").replace("TX. ", "")
        ax.set_title(f"{name}", color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=2)
        ax.text(0.99, 0.9, f"med {med:.1f}B · n={n:,}", transform=ax.transAxes,
                ha="right", va="top", color=MUTED, fontsize=8.5)
    for j in range(len(order), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Hà Nội asking-price distribution — one curve per district", color=INK,
                 fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.965, "All Phase 3 rows · districts with n≥100 · sorted by median (high→low) · dashed line = median · shared log price axis (billion ₫)",
             color=MUTED, fontsize=10.5, ha="left")
    fig.supxlabel("Asking price  (billion ₫, log scale)", color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "phase3_price_kde_by_district.png", bbox_inches="tight", facecolor=SURF)
    plt.close(fig)

    print("saved:")
    print("  figs/phase3_price_kde_by_status.png")
    print("  figs/phase3_price_kde_by_district.png")


if __name__ == "__main__":
    main()
