#!/usr/bin/env python3
"""Phase 3 — small multiples: one panel per DISTRICT, curves split by STATUS (category).

Each district panel overlays a density curve per status that has enough samples in that district
(n >= MIN_N) so the KDE is meaningful. Density on log10(price), shared log price axis.
  -> figs/phase3_price_kde_district_by_status.png
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "phase3_merged_dataset.parquet"
FIGS = pathlib.Path(__file__).resolve().parents[1] / "figs"
MIN_N = 20

BLUE, GREEN, RED, ORANGE, TEAL, PURPLE, GREY = (
    "#2a78d6", "#0a9a3c", "#c8342f", "#e08214", "#17a0a8", "#7b52c8", "#8a8f98")
INK, MUTED, GRID, SURF = "#1d2027", "#5b6675", "#e6e8ec", "#ffffff"
STATUS_COLOR = {"SOLD": GREEN, "REVOKED": ORANGE, "ON_STOCK": TEAL,
                "AVAILABLE": BLUE, "UNKNOWN": GREY, "LOCKED": PURPLE}
STATUS_ORDER = ["SOLD", "REVOKED", "ON_STOCK", "AVAILABLE", "UNKNOWN", "LOCKED"]
GRID_X = np.linspace(np.log10(0.7), np.log10(300), 400)
PX = 10 ** GRID_X


def kde(prices):
    return gaussian_kde(np.log10(prices.values))(GRID_X)


def main():
    FIGS.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA)
    df = df[df["price_bil"] > 0].copy()
    df["status"] = df["status"].replace({"ON_STOCK": "AVAILABLE"})   # merge live inventory into AVAILABLE

    vc = df["district_name"].value_counts()
    keep = vc[vc >= 100].index
    dd = df[df.district_name.isin(keep)].copy()
    order = dd.groupby("district_name")["price_bil"].median().sort_values(ascending=False).index.tolist()

    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.5, 2.7 * nrow), dpi=150, sharex=True)
    axes = axes.ravel()
    present = set()

    for i, d in enumerate(order):
        ax = axes[i]
        sub = dd[dd.district_name == d]
        ax.set_xscale("log"); ax.set_xlim(0.9, 300)
        ax.set_xticks([1, 10, 100]); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.tick_params(length=0, colors=MUTED, labelsize=9)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

        ymax = 0
        for s in STATUS_ORDER:
            g = sub.loc[sub.status == s, "price_bil"]
            if len(g) < MIN_N:
                continue
            present.add(s)
            y = kde(g); ymax = max(ymax, y.max())
            c = STATUS_COLOR[s]
            ax.fill_between(PX, y, color=c, alpha=0.07, zorder=2)
            ax.plot(PX, y, color=c, lw=2, zorder=3)
        ax.set_ylim(0, (ymax or 1) * 1.1)
        ax.set_yticks([])
        name = d.replace("Q. ", "").replace("H. ", "").replace("TX. ", "")
        med = np.median(sub["price_bil"])
        ax.set_title(name, color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=2)
        ax.text(0.99, 0.92, f"med {med:.1f}B · n={len(sub):,}", transform=ax.transAxes,
                ha="right", va="top", color=MUTED, fontsize=8.5)

    for j in range(len(order), len(axes)):
        axes[j].set_visible(False)

    handles = [Line2D([0], [0], color=STATUS_COLOR[s], lw=2.4,
                      label=f"{s}  (n={(df.status == s).sum():,})")
               for s in STATUS_ORDER if s in present]
    fig.legend(handles=handles, ncol=len(handles), frameon=False, fontsize=10, labelcolor=INK,
               loc="upper left", bbox_to_anchor=(0.02, 0.955), handlelength=1.5, columnspacing=1.6)

    fig.suptitle("Hà Nội price distribution by district, split by listing status", color=INK,
                 fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.965, f"One curve per status with n≥{MIN_N} in that district · sorted by overall median (high→low) · shared log price axis (billion ₫)",
             color=MUTED, fontsize=10.5, ha="left")
    fig.supxlabel("Asking price  (billion ₫, log scale)", color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "phase3_price_kde_district_by_status.png", bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("saved: figs/phase3_price_kde_district_by_status.png")
    print("statuses drawn:", [s for s in STATUS_ORDER if s in present])


if __name__ == "__main__":
    main()
