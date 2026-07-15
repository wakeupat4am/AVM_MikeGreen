#!/usr/bin/env python3
"""Phase 3 — price-distribution figures on the merged dataset.

  1) price distribution across ALL statuses  -> figs/phase3_price_dist_by_status.png
  2) overall price distribution divided by DISTRICT -> figs/phase3_price_dist_by_district.png

Log price axis (prices are heavily right-skewed); boxes span the IQR, whiskers the 5th-95th
percentile, the notch/line is the median. Colours are colourblind-safe.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "phase3_merged_dataset.parquet"
FIGS = pathlib.Path(__file__).resolve().parents[1] / "figs"

# colourblind-safe palette (project standard)
BLUE, GREEN, RED, ORANGE, TEAL, PURPLE, GREY = (
    "#2a78d6", "#0a9a3c", "#c8342f", "#e08214", "#17a0a8", "#7b52c8", "#8a8f98")
INK, MUTED, GRID, SURF = "#1d2027", "#5b6675", "#e6e8ec", "#ffffff"

STATUS_COLOR = {"SOLD": GREEN, "REVOKED": ORANGE, "AVAILABLE": BLUE,
                "ON_STOCK": TEAL, "LOCKED": PURPLE, "UNKNOWN": GREY}
XT = [1, 2, 5, 10, 20, 50, 100, 200]


def style(ax, title, sub):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xscale("log")
    ax.set_xticks(XT)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0.7, 300)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.tick_params(length=0, colors=MUTED, labelsize=10)
    ax.set_xlabel("Asking price  (billion ₫, log scale)", color=MUTED, fontsize=10)
    ax.set_title(title, color=INK, fontsize=15, fontweight="bold", loc="left", pad=40)
    ax.text(0, 1.035, sub, transform=ax.transAxes, color=MUTED, fontsize=10.5, va="bottom")


def boxes(ax, groups, labels, colors):
    data = [g.values for g in groups]
    pos = np.arange(len(groups))[::-1]  # first label at top
    bp = ax.boxplot(data, positions=pos, vert=False, widths=0.62, whis=(5, 95),
                    showfliers=False, patch_artist=True, medianprops=dict(color=INK, lw=2),
                    whiskerprops=dict(color=MUTED, lw=1.2), capprops=dict(color=MUTED, lw=1.2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor("white"); patch.set_linewidth(1.5)
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, color=INK, fontsize=10.5)
    for p, g in zip(pos, groups):
        med = np.median(g)
        ax.text(295, p, f"n={len(g):,}", va="center", ha="right", color=MUTED, fontsize=8.5)
        ax.text(med, p + 0.34, f"{med:.1f}", va="bottom", ha="center", color=INK, fontsize=8.5, fontweight="bold")
    return pos


def main():
    FIGS.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA)
    df = df[df["price_bil"] > 0].copy()

    # ---------- 1) by STATUS ----------
    order = df.groupby("status")["price_bil"].median().sort_values().index.tolist()
    groups = [df.loc[df.status == s, "price_bil"] for s in order]
    colors = [STATUS_COLOR.get(s, GREY) for s in order]
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=150)
    style(ax, "Hà Nội asking-price distribution by listing status",
          "Phase 3 merged set (n=27,800). Box = IQR · line = median (labelled) · whiskers = 5th–95th pct.")
    boxes(ax, groups, order, colors)
    fig.tight_layout()
    fig.savefig(FIGS / "phase3_price_dist_by_status.png", bbox_inches="tight", facecolor=SURF)
    plt.close(fig)

    # ---------- 2) by DISTRICT (all rows) ----------
    vc = df["district_name"].value_counts()
    keep = vc[vc >= 100].index
    dd = df[df.district_name.isin(keep)].copy()
    order = dd.groupby("district_name")["price_bil"].median().sort_values().index.tolist()
    groups = [dd.loc[dd.district_name == d, "price_bil"] for d in order]
    labels = [d.replace("Q. ", "").replace("H. ", "").replace("TX. ", "") for d in order]
    colors = [BLUE] * len(order)  # identity carried by the axis, not colour
    fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=150)
    style(ax, "Hà Nội asking-price distribution by district",
          f"All Phase 3 rows, districts with n≥100 ({len(order)} shown). Box = IQR · median labelled · whiskers = 5th–95th pct.")
    boxes(ax, groups, labels, colors)
    fig.tight_layout()
    fig.savefig(FIGS / "phase3_price_dist_by_district.png", bbox_inches="tight", facecolor=SURF)
    plt.close(fig)

    print("saved:")
    print("  figs/phase3_price_dist_by_status.png")
    print("  figs/phase3_price_dist_by_district.png")
    print(f"status order (low->high median): {order if False else ''}")


if __name__ == "__main__":
    main()
