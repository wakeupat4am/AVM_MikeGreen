from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCENE_ROOT = Path("/Users/dienmayhaituyet/Documents/AVM/Rule-based Scene")
SRC_DIR = SCENE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import EVAL_DIR, VIS_OUTPUT_DIR, list_metric_columns


COMBINED_DIR = EVAL_DIR / "combined"
COMBINED_DIR.mkdir(parents=True, exist_ok=True)
VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_metrics() -> pd.DataFrame:
    frames = []
    for model_name in ["baseline", "ridge", "lightgbm"]:
        path = EVAL_DIR / model_name / "all_metrics.csv"
        df = pd.read_csv(path)
        df["model_family"] = model_name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(COMBINED_DIR / "all_models_all_levels_metrics.csv", index=False)
    return combined


def save_pivot(df_level: pd.DataFrame, target_name: str, level_name: str, metric: str):
    pivot = df_level.pivot(index="group", columns="model", values=metric)
    pivot.to_csv(COMBINED_DIR / f"{target_name}_{level_name}_{metric}.csv")
    return pivot


def plot_metric_bars(pivot: pd.DataFrame, title: str, out_name: str):
    fig_height = max(4, 0.45 * len(pivot))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    pivot.plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(VIS_OUTPUT_DIR / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(df_level: pd.DataFrame, title: str, out_name: str):
    fig, ax = plt.subplots(figsize=(10, 7))
    for model_name, group_df in df_level.groupby("model"):
        ax.scatter(group_df["Median_APE"], group_df["HitRate_20_pct"], label=model_name, s=50, alpha=0.8)
        for _, row in group_df.iterrows():
            ax.annotate(str(row["group"]), (row["Median_APE"], row["HitRate_20_pct"]), fontsize=8, alpha=0.7)
    ax.set_xlabel("Median APE (%)")
    ax.set_ylabel("Hit rate within ±20%")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(VIS_OUTPUT_DIR / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bias(df_level: pd.DataFrame, title: str, out_name: str):
    pivot = df_level.pivot(index="group", columns="model", values="Bias_pct")
    fig_height = max(4, 0.45 * len(pivot))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    pivot.plot(kind="barh", ax=ax)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Bias (%)")
    plt.tight_layout()
    fig.savefig(VIS_OUTPUT_DIR / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    all_metrics = load_all_metrics()
    metric_cols = list_metric_columns()

    for target_name, df_target in all_metrics.groupby("target"):
        for level_name, df_level in df_target.groupby("level"):
            for metric in metric_cols:
                pivot = save_pivot(df_level, target_name, level_name, metric)
                plot_metric_bars(
                    pivot,
                    f"{metric} comparison | {target_name} | {level_name}",
                    f"{target_name}_{level_name}_{metric}.png",
                )

            if level_name != "overall_test_set":
                plot_tradeoff(
                    df_level,
                    f"Trade-off: Median APE vs Hit rate ±20% | {target_name} | {level_name}",
                    f"{target_name}_{level_name}_tradeoff_medianape_vs_hit20.png",
                )
                plot_bias(
                    df_level,
                    f"Bias by group | {target_name} | {level_name}",
                    f"{target_name}_{level_name}_bias.png",
                )

    print("Saved combined evaluation tables to", COMBINED_DIR)
    print("Saved visualization files to", VIS_OUTPUT_DIR)


if __name__ == "__main__":
    main()
