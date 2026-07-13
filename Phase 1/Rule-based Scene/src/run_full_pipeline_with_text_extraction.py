"""
Master runner:
  1. Load 45% of raw CSV (1,145,286 rows)
  2. Preprocess with text extraction integrated
  3. Retrain ALL existing models: baseline, ridge, lightgbm, robust_lightgbm,
     region_subtype, market_regime, address_combo
  4. Compile before/after comparison and print final results
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
MAX_RAW_ROWS = 0           # 0 = use all 914,706 rows

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


# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────

def _reload(name):
    import importlib
    mod = importlib.import_module(name)
    importlib.reload(mod)
    return mod


def _clear_argv():
    """Return a context manager that resets sys.argv to just the script name."""
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


# ─────────────────────────────────────────────
# Step 1 – process data
# ─────────────────────────────────────────────

def step_process_data():
    from common import (
        RAW_DATA_DIR, RANDOM_STATE, PROCESSED_DATA_DIR,
        build_modeling_dataset, load_raw_data, make_split,
        preprocess_raw_data, save_processed_data,
    )

    raw_csv = RAW_DATA_DIR / "mick_df_full_raw.csv"
    log("  Loading raw CSV (all rows)…")
    df_raw = load_raw_data(raw_csv)
    log(f"  Raw rows: {len(df_raw):,}")

    log("  Preprocessing (includes text extraction)…")
    df_clean = preprocess_raw_data(df_raw)
    log(f"  Clean rows: {len(df_clean):,}")

    df_model = build_modeling_dataset(df_clean)
    log(f"  Model rows: {len(df_model):,}")

    split_data = make_split(df_model)
    log(f"  Train: {len(split_data.train_df):,}  Test: {len(split_data.test_df):,}")

    save_processed_data(df_clean, df_model, split_data)
    log(f"  Saved to {PROCESSED_DATA_DIR}")


# ─────────────────────────────────────────────
# Steps 2-8 – train models
# ─────────────────────────────────────────────

def step_baseline():
    _reload("train_baseline").main()


def step_ridge():
    _reload("train_ridge").main()


def step_lightgbm():
    _reload("train_lightgbm").main()


def step_robust_lightgbm():
    _reload("train_robust_lightgbm").main()


def step_region_subtype():
    with _clear_argv():
        _reload("train_region_subtype_models").main()


def step_market_regime():
    with _clear_argv():
        _reload("train_market_regime_models").main()


def step_address_combo():
    _reload("train_address_combo_comparison").main()


# ─────────────────────────────────────────────
# Step 9 – compile before / after comparison
# ─────────────────────────────────────────────

def step_compile_results():
    import pandas as pd

    before_path = EVAL_DIR / "improvements" / "before_text_extraction_snapshot.csv"
    before = pd.read_csv(before_path) if before_path.exists() else pd.DataFrame()
    if not before.empty:
        before = before.rename(columns={
            "Median_APE":    "Median_APE_before",
            "HitRate_20_pct": "HitRate_20_before",
        })

    rows = []

    def collect(exp_label, folder):
        for p in sorted(Path(folder).rglob("overall_test_set.csv")):
            if "price_per_m2" in str(p):
                continue
            try:
                df = pd.read_csv(p)
                if "Median_APE" not in df.columns:
                    continue
                for _, r in df.iterrows():
                    rows.append({
                        "experiment":      exp_label,
                        "model":           str(r.get("model", p.parent.name)),
                        "Median_APE_after":   round(float(r["Median_APE"]), 4),
                        "HitRate_20_after":   round(float(r["HitRate_20_pct"]), 4),
                        "HitRate_10_after":   round(float(r.get("HitRate_10_pct", float("nan"))), 4),
                        "MAPE_after":         round(float(r.get("MAPE", float("nan"))), 4),
                    })
            except Exception:
                pass

    collect("baseline",              EVAL_DIR / "baseline")
    collect("ridge",                 EVAL_DIR / "ridge")
    collect("lightgbm",              EVAL_DIR / "lightgbm")
    collect("robust_lightgbm",       EVAL_DIR / "improvements/robust_lightgbm")
    collect("region_subtype_models", EVAL_DIR / "improvements/region_subtype_models")
    collect("market_regime_layer",   EVAL_DIR / "improvements/market_regime_layer")
    collect("address_combo",         EVAL_DIR / "improvements/address_combo_comparison")

    after = pd.DataFrame(rows).drop_duplicates(["experiment", "model"])

    if not before.empty:
        merged = after.merge(
            before[["experiment", "model", "Median_APE_before", "HitRate_20_before"]],
            on=["experiment", "model"], how="left",
        )
        merged["Δ_Median_APE"]   = (merged["Median_APE_after"]  - merged["Median_APE_before"]).round(4)
        merged["Δ_HitRate_20"]   = (merged["HitRate_20_after"]  - merged["HitRate_20_before"]).round(4)
    else:
        merged = after

    merged = merged.sort_values("Median_APE_after")
    out = EVAL_DIR / "improvements" / "text_extraction_final_comparison.csv"
    merged.to_csv(out, index=False)

    # ── print summary ──────────────────────────────────────────────────
    print()
    print("=" * 110)
    print("FINAL RESULTS — ALL MODELS RETRAINED ON 45% DATASET WITH TEXT EXTRACTION")
    print("=" * 110)

    show_cols = [c for c in [
        "experiment", "model",
        "Median_APE_before", "Median_APE_after", "Δ_Median_APE",
        "HitRate_20_before", "HitRate_20_after", "Δ_HitRate_20",
    ] if c in merged.columns]
    print(merged[show_cols].to_string(index=False))
    print("=" * 110)

    # ── highlight best model ──
    best = merged.loc[merged["Median_APE_after"].idxmin()]
    print(f"\nBest model: [{best['experiment']}]  {best['model']}")
    print(f"  Median_APE : {best['Median_APE_after']:.4f}%", end="")
    if "Δ_Median_APE" in best:
        sign = "↓" if best["Δ_Median_APE"] < 0 else "↑"
        print(f"  ({sign}{abs(best['Δ_Median_APE']):.4f} vs before)", end="")
    print()
    print(f"  HitRate@20 : {best['HitRate_20_after']:.4f}%", end="")
    if "Δ_HitRate_20" in best:
        sign = "↑" if best["Δ_HitRate_20"] > 0 else "↓"
        print(f"  ({sign}{abs(best['Δ_HitRate_20']):.4f} vs before)", end="")
    print()
    print(f"\nSaved to: {out}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

STEPS = [
    ("1_process_data    (45% of raw, +text extraction)",  step_process_data),
    ("2_baseline",                                         step_baseline),
    ("3_ridge",                                            step_ridge),
    ("4_lightgbm",                                         step_lightgbm),
    ("5_robust_lightgbm",                                  step_robust_lightgbm),
    ("6_region_subtype  (global/region/subtype/hybrid)",   step_region_subtype),
    ("7_market_regime",                                    step_market_regime),
    ("8_address_combo",                                    step_address_combo),
    ("9_compile_results",                                  step_compile_results),
]

if __name__ == "__main__":
    log("Pipeline start — using ALL 914,706 raw rows (100%)")
    for label, fn in STEPS:
        run_step(label, fn)
    log("ALL STEPS COMPLETE")
