#!/usr/bin/env python3
"""Phase 3 — Hà Nội price time series from last_modified_date across ALL dataset members.

Top panel : monthly median asking price/m² (million ₫/m²) with P25-P75 band, plus the trend
            split by source (transaction vs listing) where each month has >=15 rows.
Bottom    : monthly row count, stacked by source (shows coverage + composition honestly).

Only rows with a real last_modified date are placed on the axis (undated Phase-1 listings can't be).
  -> figs/phase3_price_timeseries.png
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "phase3_merged_dataset.parquet"
FIGS = pathlib.Path(__file__).resolve().parents[1] / "figs"

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#e08214", "#0a9a3c", "#8a8f98"
INK, MUTED, GRID, SURF, BAND = "#1d2027", "#5b6675", "#e6e8ec", "#ffffff", "#c9ddf6"
MINN_LINE = 15   # min rows/month to plot a source-split point


def monthly_median(df):
    g = df.groupby("ym")["ppm2_m"]
    out = pd.DataFrame({"med": g.median(), "p25": g.quantile(.25), "p75": g.quantile(.75), "n": g.size()})
    out.index = out.index.to_timestamp()
    return out


def main():
    FIGS.mkdir(exist_ok=True)
    d = pd.read_parquet(DATA)
    d["lm"] = pd.to_datetime(d["last_modified_date"], errors="coerce")
    d = d[d["lm"].notna() & (~d["suspicious_price"]) & (d["price_per_m2"] > 0)].copy()
    d["ppm2_m"] = d["price_per_m2"] / 1e6            # -> million ₫/m²
    d["ym"] = d["lm"].dt.to_period("M")

    overall = monthly_median(d)
    overall = overall[overall.n >= 10]
    src = {s: monthly_median(g) for s, g in d.groupby("dataset_source")}

    fig, (ax, axb) = plt.subplots(2, 1, figsize=(12, 7.2), dpi=150, sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1], hspace=0.12))

    # ---- top: median price/m² ----
    ax.fill_between(overall.index, overall.p25, overall.p75, color=BAND, alpha=0.5, lw=0,
                    zorder=1, label="P25–P75 (all members)")
    ax.plot(overall.index, overall.med, color=INK, lw=2.6, zorder=4, label="Median — all members")
    for s, c, lab in [("phase2_transaction", BLUE, "Transaction pool"), ("phase1_listing", ORANGE, "Scraped listings")]:
        t = src[s][src[s].n >= MINN_LINE]
        ax.plot(t.index, t.med, color=c, lw=1.8, ls=(0, (4, 2)), marker="o", ms=3.5, zorder=3, label=lab)
    ax.set_ylabel("Median asking price / m²  (million ₫)", color=MUTED, fontsize=10.5)
    ax.set_ylim(0, None)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=MUTED, labelsize=9.5)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=INK, ncol=2, columnspacing=1.4)
    ax.set_title("Hà Nội asking price/m² over time — all Phase 3 members (by last-modified date)",
                 color=INK, fontsize=15, fontweight="bold", loc="left", pad=34)
    ax.text(0, 1.045, "Monthly median with P25–P75 band · dashed = split by source (months with n≥15) · undated listings excluded",
            transform=ax.transAxes, color=MUTED, fontsize=10.5, va="bottom")

    # ---- bottom: monthly counts stacked by source ----
    piv = d.pivot_table(index=d.ym.dt.to_timestamp(), columns="dataset_source", values="lm",
                        aggfunc="size", fill_value=0)
    x = piv.index
    tx = piv.get("phase2_transaction", pd.Series(0, index=x))
    ls = piv.get("phase1_listing", pd.Series(0, index=x))
    w = 20
    axb.bar(x, tx, width=w, color=BLUE, label="Transaction pool", zorder=3)
    axb.bar(x, ls, width=w, bottom=tx, color=ORANGE, label="Scraped listings", zorder=3)
    axb.set_ylabel("rows / month", color=MUTED, fontsize=10)
    axb.spines["top"].set_visible(False); axb.spines["right"].set_visible(False)
    axb.spines["left"].set_color(GRID); axb.spines["bottom"].set_color(GRID)
    axb.yaxis.grid(True, color=GRID, lw=1); axb.set_axisbelow(True)
    axb.tick_params(length=0, colors=MUTED, labelsize=9.5)
    axb.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK, ncol=2)
    axb.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axb.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    fig.savefig(FIGS / "phase3_price_timeseries.png", bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("saved: figs/phase3_price_timeseries.png")
    print("span:", overall.index.min().date(), "->", overall.index.max().date(),
          "| dated rows used:", len(d))


if __name__ == "__main__":
    main()
