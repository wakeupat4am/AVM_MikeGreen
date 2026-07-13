"""
CatBoost pipeline: trains only region_subtype + market_regime models with CatBoost.
Reuses the already-processed parquet files — no re-processing of raw data.

Outputs are compared against the LightGBM equivalents as the "before" baseline.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

SCENE_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = SCENE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.chdir(SRC_DIR)

EVAL_DIR = SCENE_ROOT / "Evaluation"
_START = time.time()


def log(msg: str):
    elapsed = time.time() - _START
    m, s = divmod(int(elapsed), 60)
    print(f"[{m:02d}:{s:02d}] {msg}", flush=True)


def run_step(label: str, fn):
    log(f"START  {label}")
    t0 = time.time()
    try:
        fn()
        log(f"DONE   {label}  ({time.time()-t0:.0f}s)")
    except Exception:
        log(f"ERROR  {label}")
        traceback.print_exc()
        raise


def _reload(name):
    import importlib
    mod = importlib.import_module(name)
    importlib.reload(mod)
    return mod


def _clear_argv():
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        orig = sys.argv[:]
        sys.argv = [sys.argv[0]]
        try:
            yield
        finally:
            sys.argv = orig

    return _ctx()


# ── steps ─────────────────────────────────────────────────────────────────────

def step_catboost_region_subtype():
    with _clear_argv():
        _reload("train_catboost_region_subtype").main()


def step_catboost_market_regime():
    with _clear_argv():
        _reload("train_catboost_market_regime").main()


def step_compile_results():
    import pandas as pd

    # LightGBM run as "before" baseline
    lgbm_region_subtype_path = EVAL_DIR / "improvements" / "region_subtype_models"
    lgbm_market_regime_path  = EVAL_DIR / "improvements" / "market_regime_layer"

    rows_before = []
    rows_after  = []

    def collect(label, folder, target_rows):
        for p in sorted(Path(folder).rglob("overall_test_set.csv")):
            if "price_per_m2" in str(p):
                continue
            try:
                df = pd.read_csv(p)
                if "Median_APE" not in df.columns:
                    continue
                for _, r in df.iterrows():
                    target_rows.append({
                        "experiment": label,
                        "model": str(r.get("model", p.parent.name)),
                        "Median_APE": round(float(r["Median_APE"]), 4),
                        "HitRate_20_pct": round(float(r["HitRate_20_pct"]), 4),
                        "HitRate_10_pct": round(float(r.get("HitRate_10_pct", float("nan"))), 4),
                        "MAPE": round(float(r.get("MAPE", float("nan"))), 4),
                    })
            except Exception:
                pass

    collect("lgbm_region_subtype", lgbm_region_subtype_path, rows_before)
    collect("lgbm_market_regime",  lgbm_market_regime_path,  rows_before)
    collect("catboost_region_subtype", EVAL_DIR / "improvements" / "catboost_region_subtype_models", rows_after)
    collect("catboost_market_regime",  EVAL_DIR / "improvements" / "catboost_market_regime_layer",   rows_after)

    before = pd.DataFrame(rows_before).drop_duplicates(["experiment", "model"])
    after  = pd.DataFrame(rows_after).drop_duplicates(["experiment", "model"])

    # Strip "lgbm_" / "catboost_" prefix to align by model family + model name
    before["family"] = before["experiment"].str.replace("lgbm_", "", regex=False)
    after["family"]  = after["experiment"].str.replace("catboost_", "", regex=False)

    merged = after.merge(
        before[["family", "model", "Median_APE", "HitRate_20_pct"]].rename(
            columns={"Median_APE": "Median_APE_lgbm", "HitRate_20_pct": "HitRate_20_lgbm"}
        ),
        on=["family", "model"], how="left",
    )
    merged["Δ_Median_APE"]  = (merged["Median_APE"]      - merged["Median_APE_lgbm"]).round(4)
    merged["Δ_HitRate_20"]  = (merged["HitRate_20_pct"]  - merged["HitRate_20_lgbm"]).round(4)

    # Focus: region subtype model, hybrid fallback, probability adjustment
    focus_models = {"region subtype model", "hybrid fallback", "probability adjustment"}
    focus = merged[merged["model"].isin(focus_models)].copy()
    full  = merged.sort_values("Median_APE")

    out = EVAL_DIR / "improvements" / "catboost_final_comparison.csv"
    full.to_csv(out, index=False)

    print()
    print("=" * 120)
    print("CATBOOST vs LIGHTGBM — FOCUSED MODELS")
    print("=" * 120)
    show_cols = [c for c in ["experiment", "model", "Median_APE_lgbm", "Median_APE", "Δ_Median_APE",
                               "HitRate_20_lgbm", "HitRate_20_pct", "Δ_HitRate_20"] if c in focus.columns]
    print(focus.sort_values("Median_APE")[show_cols].to_string(index=False))
    print()
    print("=" * 120)
    print("ALL MODELS")
    print("=" * 120)
    print(full[show_cols].to_string(index=False))
    print("=" * 120)

    best = full.loc[full["Median_APE"].idxmin()]
    print(f"\nBest model : [{best['experiment']}]  {best['model']}")
    print(f"  Median_APE : {best['Median_APE']:.4f}%", end="")
    if pd.notna(best.get("Δ_Median_APE")):
        d = best["Δ_Median_APE"]
        print(f"  ({'↓' if d < 0 else '↑'}{abs(d):.4f} vs LightGBM)", end="")
    print()
    print(f"  HitRate@20 : {best['HitRate_20_pct']:.4f}%", end="")
    if pd.notna(best.get("Δ_HitRate_20")):
        d = best["Δ_HitRate_20"]
        print(f"  ({'↑' if d > 0 else '↓'}{abs(d):.4f} vs LightGBM)", end="")
    print()
    log(f"Saved to {out}")


# ── main ──────────────────────────────────────────────────────────────────────

STEPS = [
    ("1_catboost_region_subtype", step_catboost_region_subtype),
    ("2_catboost_market_regime",  step_catboost_market_regime),
    ("3_compile_results",         step_compile_results),
]

if __name__ == "__main__":
    log("CatBoost pipeline start")
    for label, fn in STEPS:
        run_step(label, fn)
    log("ALL STEPS COMPLETE")
