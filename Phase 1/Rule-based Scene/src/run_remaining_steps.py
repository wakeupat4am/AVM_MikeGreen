"""
Resume from step 7 — steps 1-6 already completed successfully.
Runs: market_regime, address_combo, compile_results.
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


def run_step(label, fn):
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


def step_market_regime():
    with _clear_argv():
        _reload("train_market_regime_models").main()


def step_address_combo():
    _reload("train_address_combo_comparison").main()


def step_compile_results():
    import pandas as pd

    before_path = EVAL_DIR / "improvements" / "before_text_extraction_snapshot.csv"
    before = pd.read_csv(before_path) if before_path.exists() else pd.DataFrame()
    if not before.empty:
        before = before.rename(columns={
            "Median_APE":     "Median_APE_before",
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
                        "experiment":       exp_label,
                        "model":            str(r.get("model", p.parent.name)),
                        "Median_APE_after": round(float(r["Median_APE"]), 4),
                        "HitRate_20_after": round(float(r["HitRate_20_pct"]), 4),
                        "HitRate_10_after": round(float(r.get("HitRate_10_pct", float("nan"))), 4),
                        "MAPE_after":       round(float(r.get("MAPE", float("nan"))), 4),
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
        merged["Δ_Median_APE"] = (merged["Median_APE_after"] - merged["Median_APE_before"]).round(4)
        merged["Δ_HitRate_20"] = (merged["HitRate_20_after"] - merged["HitRate_20_before"]).round(4)
    else:
        merged = after

    merged = merged.sort_values("Median_APE_after")
    out = EVAL_DIR / "improvements" / "text_extraction_final_comparison.csv"
    merged.to_csv(out, index=False)

    print()
    print("=" * 110)
    print("FINAL RESULTS — ALL MODELS RETRAINED ON 100% DATASET WITH TEXT EXTRACTION")
    print("=" * 110)
    show_cols = [c for c in [
        "experiment", "model",
        "Median_APE_before", "Median_APE_after", "Δ_Median_APE",
        "HitRate_20_before", "HitRate_20_after", "Δ_HitRate_20",
    ] if c in merged.columns]
    print(merged[show_cols].to_string(index=False))
    print("=" * 110)

    best = merged.loc[merged["Median_APE_after"].idxmin()]
    print(f"\nBest model : [{best['experiment']}]  {best['model']}")
    print(f"  Median_APE : {best['Median_APE_after']:.4f}%", end="")
    if "Δ_Median_APE" in merged.columns:
        delta = best["Δ_Median_APE"]
        print(f"  ({'↓' if delta < 0 else '↑'}{abs(delta):.4f} vs before)", end="")
    print()
    print(f"  HitRate@20 : {best['HitRate_20_after']:.4f}%", end="")
    if "Δ_HitRate_20" in merged.columns:
        delta = best["Δ_HitRate_20"]
        print(f"  ({'↑' if delta > 0 else '↓'}{abs(delta):.4f} vs before)", end="")
    print()
    log(f"Saved to {out}")


STEPS = [
    ("7_market_regime",  step_market_regime),
    ("8_address_combo",  step_address_combo),
    ("9_compile_results", step_compile_results),
]

if __name__ == "__main__":
    log("Resuming from step 7 (steps 1-6 already done)")
    for label, fn in STEPS:
        run_step(label, fn)
    log("ALL STEPS COMPLETE")
