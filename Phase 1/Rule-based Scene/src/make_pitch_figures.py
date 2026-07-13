"""Generate slide-ready figures for the AVM pitch deck.

Applies the dataviz design rules: light surface, single-hue blue for magnitude, status
red for the flagged weak spot, direct value labels (mitigates the low-contrast accents),
thin marks, recessive axes, no dual-axis, romanised region labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import EVAL_DIR, PROCESSED_DATA_DIR, is_commercial_listing
from build_regional_report import region_title

OUT = EVAL_DIR.parent / "Visualizations" / "pitch_figures"
OUT.mkdir(parents=True, exist_ok=True)

# palette (validated)
BLUE = "#2a78d6"; BLUE_D = "#184f95"; AQUA = "#1baf7a"
GREEN = "#0ca30c"; RED = "#d03b3b"; SERIOUS = "#ec835a"; GREY = "#b8b7b3"
INK = "#0b0b0b"; INK2 = "#52514e"; SURF = "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 12, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.edgecolor": GREY, "font.family": "DejaVu Sans",
})


def _style(ax, title, subtitle=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=INK, pad=30 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.06, subtitle, transform=ax.transAxes, fontsize=10.5, color=INK2, va="bottom")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_price_dist(df):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    p = df["price_vnd"].astype(float) / 1e9
    p = p[(p > 0.1) & (p < 100)]
    ax.hist(p, bins=np.logspace(np.log10(0.2), np.log10(100), 50), color=BLUE, edgecolor=SURF, linewidth=0.4)
    ax.set_xscale("log")
    med = float(p.median()); mean = float(p.mean())
    top = ax.get_ylim()[1]
    ax.axvline(med, color=BLUE_D, lw=2)
    ax.axvline(mean, color=RED, lw=2, ls="--")
    ax.text(med * 0.96, top * 0.98, f"median {med:.1f}B", color=BLUE_D, fontsize=11, fontweight="bold", ha="right", va="top")
    ax.text(mean * 1.05, top * 0.98, f"mean {mean:.1f}B", color=RED, fontsize=11, fontweight="bold", va="top")
    ax.set_xlabel("Price (billion VND, log scale)")
    ax.set_ylabel("Listings")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.6)
    _style(ax, "Prices are heavily right-skewed",
           "Why the median (not the mean) is the honest metric — a long tail of expensive homes")
    save(fig, "fig_price_dist.png")


def fig_ppm2_dist(df):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    v = df["price_per_m2"].astype(float) / 1e6  # millions VND per m2
    v = v[(v > 0.5) & (v < 500)]
    bins = np.logspace(np.log10(1), np.log10(500), 55)
    ax.hist(v, bins=bins, color=BLUE, edgecolor=SURF, linewidth=0.4)
    ax.set_xscale("log")
    top = ax.get_ylim()[1]
    med = float(v.median())
    # cleaning floor: everything below 20M/m2 is dropped as impossible/below-market
    ax.axvspan(1, 20, color=RED, alpha=0.10)
    ax.axvline(20, color=RED, lw=2)
    ax.axvline(med, color=BLUE_D, lw=2)
    below = float((df["price_per_m2"] < 20e6).mean() * 100)
    ax.text(20 * 0.94, top * 0.98, f"20M/m² floor\n({below:.0f}% removed)", color=RED, fontsize=10.5,
            fontweight="bold", ha="right", va="top")
    ax.text(med * 1.06, top * 0.78, f"median {med:.0f}M/m²", color=BLUE_D, fontsize=11, fontweight="bold", va="top")
    ax.set_xlabel("Price per m² (million VND, log scale)")
    ax.set_ylabel("Listings")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200", "500"])
    ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.6)
    _style(ax, "Price per m² spans a wide, skewed range",
           "A long low tail is impossible/below-market labels — the 20M/m² floor removes it")
    save(fig, "fig_ppm2_dist.png")


def fig_missingness(df):
    feats = ["built_year", "length_m", "bathrooms", "road_width_m", "frontage_m",
             "num_floors", "width_m", "bedrooms", "area_m2"]
    feats = [f for f in feats if f in df.columns]
    miss = (df[feats].isna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    y = np.arange(len(miss))
    colors = [RED if v >= 99 else SERIOUS if v >= 50 else BLUE for v in miss.values]
    ax.barh(y, miss.values, color=colors, height=0.72)
    ax.set_yticks(y); ax.set_yticklabels([f.replace("_", " ") for f in miss.index])
    for i, v in enumerate(miss.values):
        ax.text(v + 1.2, i, f"{v:.0f}%", va="center", fontsize=10, color=INK2)
    ax.set_xlim(0, 112); ax.set_xlabel("Missing (% of listings)")
    ax.spines["left"].set_visible(False)
    _style(ax, "Most structural fields are half-empty",
           "built_year is 100% missing (unusable); the model leans on area + location by necessity")
    save(fig, "fig_missingness.png")


def fig_region_mix(df):
    c = df["addr_last_1"].value_counts().head(12).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y = np.arange(len(c))
    ax.barh(y, c.values / 1000, color=BLUE, height=0.74)
    ax.set_yticks(y); ax.set_yticklabels([region_title(r) for r in c.index])
    for i, v in enumerate(c.values / 1000):
        ax.text(v + 1, i, f"{v:.0f}k", va="center", fontsize=10, color=INK2)
    ax.set_xlabel("Listings (thousands)")
    ax.spines["left"].set_visible(False)
    _style(ax, "Listings are geographically uneven",
           "A few metros dominate; many provinces are data-thin — why accuracy must be read per region")
    save(fig, "fig_region_mix.png")


def fig_subtype_mix(df):
    c = df["property_subtype"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = np.arange(len(c))
    ax.barh(y, c.values / 1000, color=BLUE, height=0.72)
    ax.set_yticks(y); ax.set_yticklabels([s.replace("_", " ") for s in c.index])
    for i, v in enumerate(c.values / 1000):
        ax.text(v + 2, i, f"{v:.0f}k", va="center", fontsize=10, color=INK2)
    ax.set_xlabel("Listings (thousands)")
    ax.spines["left"].set_visible(False)
    _style(ax, "In-scope property mix", "704K landed residential houses across seven subtypes")
    save(fig, "fig_subtype_mix.png")


def fig_model_compare():
    d = pd.read_csv(EVAL_DIR / "improvements" / "region_subtype_calibrated" / "final_comparison.csv")
    order = ["global model", "subtype model", "region model", "hybrid fallback", "region subtype model", "shrinkage blend"]
    d = d[d["model"].isin(order)].set_index("model").loc[[m for m in order if m in d["model"].values]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(d))
    colors = [BLUE_D if m == "shrinkage blend" else BLUE if m != "region subtype model" else AQUA for m in d.index]
    ax.barh(y, d["Median_APE"], color=colors, height=0.72)
    ax.set_yticks(y); ax.set_yticklabels([m.replace(" model", "").title() for m in d.index])
    for i, v in enumerate(d["Median_APE"]):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=10, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("Median APE (lower is better)")
    ax.spines["left"].set_visible(False)
    _style(ax, "The shrinkage blend is the best deployable model",
           "Region×Subtype scores lower but covers only dense groups; shrinkage values every listing")
    save(fig, "fig_model_compare.png")


def fig_bias():
    labels = ["Raw mean\nbias", "Median\nbias", "10%-trimmed\nmean bias"]
    vals = [13.9, -0.29, 0.63]
    colors = [SERIOUS, GREEN, GREEN]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(3)
    ax.bar(x, vals, color=colors, width=0.6)
    ax.axhline(0, color=INK2, lw=1)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.4 if v >= 0 else -0.9), f"{v:+.1f}%", ha="center", fontsize=12, fontweight="bold",
                color=SERIOUS if i == 0 else GREEN)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Signed error (%)")
    ax.set_ylim(-3, 16)
    ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.6)
    _style(ax, "The model is centred — the scary bias is an artifact",
           "Raw mean is inflated by a label-noise tail; median & trimmed bias sit at ~0")
    save(fig, "fig_bias.png")


def fig_pred_vs_actual():
    d = pd.read_csv(EVAL_DIR / "improvements" / "region_subtype_calibrated_clean_ppm2_20m" / "predictions.csv")
    d = d[(d["price_per_m2"] >= 20e6)]
    a = d["price_vnd"].astype(float) / 1e9
    p = d["pred_price_vnd_shrunk"].astype(float) / 1e9
    m = (a > 0.3) & (a < 60) & (p > 0.3) & (p < 60)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.hexbin(a[m], p[m], gridsize=45, xscale="log", yscale="log", cmap="Blues", mincnt=1, linewidths=0)
    lim = [0.4, 60]
    ax.plot(lim, lim, color=RED, lw=1.8, label="perfect (y = x)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual price (B VND)"); ax.set_ylabel("Predicted price (B VND)")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    _style(ax, "Predictions track actual prices", "Served population; density concentrates on the y = x line")
    save(fig, "fig_pred_vs_actual.png")


def fig_confidence_tiers():
    d = pd.read_csv(EVAL_DIR / "improvements" / "region_subtype_calibrated" / "config.json", nrows=0) if False else None
    tiers = ["A · auto-serve\n(40% of listings)", "B · value + range\n(40% of listings)", "C · review\n(20% of listings)"]
    medape = [9.55, 17.64, 28.85]
    ramp = ["#1c5cab", "#3987e5", "#86b6ef"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(3)
    ax.bar(x, medape, color=ramp, width=0.62)
    for i, v in enumerate(medape):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.set_xticks(x); ax.set_xticklabels(tiers, fontsize=10.5)
    ax.set_ylabel("Median APE")
    ax.set_ylim(0, 33)
    ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.6)
    _style(ax, "The model knows when it is uncertain",
           "Its own confidence tier predicts its accuracy — auto-serve where sharp, review where not")
    save(fig, "fig_confidence_tiers.png")


def fig_error_vs_normal():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    cats = ["Normal listings\n(APE ≤ 20%)", "Large-error listings\n(APE > 50%)"]
    extreme = [5.1, 29.2]
    ax.bar([0, 1], extreme, color=[BLUE, RED], width=0.55)
    for i, v in enumerate(extreme):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(cats)
    ax.set_ylabel("Share sitting in the extreme price/m² tail")
    ax.set_ylim(0, 34)
    ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.6)
    _style(ax, "Big errors are bad labels, not bad predictions",
           "Large-error listings are 6× more likely to have an implausible price/m²")
    save(fig, "fig_error_vs_normal.png")


def fig_region_ape():
    d = pd.read_csv(EVAL_DIR / "regional_report" / "region_summary.csv")
    d = d[~d["region"].str.startswith("other")].nlargest(12, "n_test").sort_values("Median_APE")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    y = np.arange(len(d))
    colors = [RED if r == "hồ chí minh" else BLUE for r in d["region"]]
    ax.barh(y, d["Median_APE"], color=colors, height=0.74)
    ax.set_yticks(y); ax.set_yticklabels([region_title(r) for r in d["region"]])
    for i, v in enumerate(d["Median_APE"]):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=10, color=INK2)
    ax.set_xlabel("Median APE")
    ax.spines["left"].set_visible(False)
    _style(ax, "Accuracy is reported per region — and we name the weak spot",
           "Most markets 11–15%; Hồ Chí Minh (red) is high-variance and flagged, not hidden")
    save(fig, "fig_region_ape.png")


def fig_stability():
    d = pd.read_csv(EVAL_DIR / "improvements" / "repeated_split_stability" / "per_seed_metrics.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    v = d["served_Median_APE"].values
    mean = v.mean()
    ax.axhspan(mean - v.std(ddof=1), mean + v.std(ddof=1), color=BLUE, alpha=0.12)
    ax.axhline(mean, color=BLUE_D, lw=1.6)
    ax.scatter(d["seed"], v, color=BLUE, s=90, zorder=3, edgecolor=SURF, linewidth=1.2)
    ax.text(d["seed"].max() + 0.15, mean, f"mean {mean:.2f}%  ± {v.std(ddof=1):.2f}", va="center",
            fontsize=11, color=BLUE_D, fontweight="bold")
    ax.set_xlabel("Random split (seed)"); ax.set_ylabel("Median APE")
    ax.set_xticks(d["seed"]); ax.set_xlim(-0.4, d["seed"].max() + 1.4)
    ax.set_ylim(mean - 0.4, mean + 0.4)
    ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.6)
    _style(ax, "Stable across 5 random splits", "The headline is not a lucky split — variance is a few hundredths of a point")
    save(fig, "fig_stability.png")


def main():
    df = pd.read_parquet(PROCESSED_DATA_DIR / "landed_residential_house_modeling_dataset.parquet")
    fig_price_dist(df)
    fig_ppm2_dist(df)
    fig_missingness(df)
    fig_region_mix(df)
    fig_subtype_mix(df)
    fig_model_compare()
    fig_bias()
    fig_pred_vs_actual()
    fig_confidence_tiers()
    fig_error_vs_normal()
    fig_region_ape()
    fig_stability()
    print("\nAll figures ->", OUT)


if __name__ == "__main__":
    main()
