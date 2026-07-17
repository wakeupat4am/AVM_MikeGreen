#!/usr/bin/env python3
"""Phase 3 — prediction intervals v2: drift-proof CQR + hedonic market index.

Fixes the two weaknesses the rolling backtest exposed in v1 (train_phase3_intervals.py):

  1. Index: v1 smoothed the monthly median with a *centered* window (sees the future) and the
     raw median confounds market moves with composition shifts (transactions early, scraped
     listings late). v2 uses a **hedonic index** — month coefficients of
     log(price/m2) ~ month + district + log(area) + floors + facade — which is causal per month
     and composition-controlled. It was also the best backtest variant (36.3% vs 43.9% MedAPE
     at the hardest origin).

  2. Calibration: v1's random-fold CQR promised 80% coverage but delivered 67–77% in real time.
     v2 calibrates **temporally** (a trailing window of the most recent training months, grown
     until >= MIN_CAL rows) at **nominal 90%** — the drift safety margin backtest-selected in
     backtest_cqr_fix.py: real-time coverage 81.6–95.5% at every origin, never below the 80%
     promise. Models are fit strictly before the calibration window (split conformal).

Cost of honesty: median band width x~3.6 (was x2.19). That is the true uncertainty under
drift; the old width was a random-fold artifact.

Output: phase3_available_intervals_v2.csv, phase3_unknown_intervals_v2.csv.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from build_phase3_modeling import load_splits, FEATS, TARGET
from train_phase3_intervals import make_q
from backtest_rolling_origin import load_market_rows, hedonic_index_path
from backtest_cqr_fix import wquantile

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
NOMINAL = 0.90          # backtest-selected drift margin -> delivers >= 80% real coverage
MIN_CAL = 150           # grow the trailing calibration window until this many rows
REF = pd.Period("2024-06", "M")


def hedonic_index():
    """Full-fit hedonic index {month -> level} (latest cutoff of the causal path)."""
    path = hedonic_index_path(load_market_rows())
    return path[max(path)]


def index_scale(dates, hidx, ref):
    """I(t)/I(ref) per row; undated -> 1.0 (reference level); months clamped to the index range."""
    per = pd.to_datetime(dates, errors="coerce").dt.to_period("M")
    lo, hi = min(hidx), max(hidx)
    out = np.ones(len(per))
    for i, p in enumerate(per):
        if pd.isna(p):
            continue
        p = min(max(p, lo), hi)
        while p not in hidx and p > lo:     # skip gap months
            p -= 1
        out[i] = hidx[p] / hidx[ref]
    return out


def main():
    hidx = hedonic_index()
    print(f"hedonic index: {min(hidx)}..{max(hidx)}  "
          f"rise x{max(hidx.values()) / hidx[min(hidx)]:.2f} vs first month")

    train, av, un = load_splits()
    train["month"] = pd.to_datetime(train["last_modified_date"], errors="coerce").dt.to_period("M")
    train = train[train["month"].notna()].copy()
    ly = np.log1p(train[TARGET].values / index_scale(train["last_modified_date"], hidx, REF))

    # trailing temporal calibration window (most recent months, >= MIN_CAL rows)
    months = sorted(train["month"].unique(), reverse=True)
    cal_start, n = None, 0
    for m in months:
        n += int((train["month"] == m).sum())
        cal_start = m
        if n >= MIN_CAL:
            break
    cal_m = (train["month"] >= cal_start).values
    fit_m = ~cal_m
    print(f"calibration window: {cal_start}..{months[0]}  n_cal={cal_m.sum()}  n_fit={fit_m.sum():,}")

    models = {a: make_q(a).fit(train[fit_m][FEATS], ly[fit_m],
                               m__sample_weight=train[fit_m]["w"].values) for a in (0.10, 0.50, 0.90)}
    l10c = models[0.10].predict(train[cal_m][FEATS])
    l90c = models[0.90].predict(train[cal_m][FEATS])
    E = np.maximum(l10c - ly[cal_m], ly[cal_m] - l90c)
    Q = wquantile(E, np.ones(len(E)), NOMINAL)
    print(f"conformity Q = {Q:.3f} (log space, nominal {NOMINAL:.0%}) -> band widening x{np.exp(Q):.2f}")

    def score(df, tag, fn):
        sc = index_scale(df["last_modified_date"], hidx, REF)
        l10 = models[0.10].predict(df[FEATS]); l50 = models[0.50].predict(df[FEATS]); l90 = models[0.90].predict(df[FEATS])
        p10 = np.expm1(l10 - Q) * sc; p50 = np.expm1(l50) * sc; p90 = np.expm1(l90 + Q) * sc
        ask = df[TARGET].values
        flag = np.where(ask < p10, "UNDERPRICED",
               np.where(ask > p90 * 1.3, "STRONGLY_OVER",
               np.where(ask > p90, "OVERPRICED", "fair")))
        out = df[["property_id", "dataset_source", "district_name", "ward_name", "total_area",
                  "number_of_floors", "last_modified_date"]].copy()
        out["asking_B"] = (ask / 1e9).round(2)
        out["P10_B"] = (p10 / 1e9).round(2); out["P50_B"] = (p50 / 1e9).round(2); out["P90_B"] = (p90 / 1e9).round(2)
        out["over_P90_pct"] = ((ask - p90) / p90 * 100).round(1)
        out["flag"] = flag
        out["time_adjusted"] = pd.to_datetime(df["last_modified_date"], errors="coerce").notna().values
        out.to_csv(DATA / fn, index=False)
        vc = pd.Series(flag).value_counts()
        print(f"\n=== SCORE v2 (drift-proof bands): {tag}  (n={len(df):,}) -> {fn} ===")
        print(f"  median band width P90/P10: x{np.median(p90 / np.maximum(p10, 1)):.2f}")
        for k in ["UNDERPRICED", "fair", "OVERPRICED", "STRONGLY_OVER"]:
            c = int(vc.get(k, 0)); print(f"    {k:14s} {c:6,d}  {c/len(df)*100:5.1f}%")

    score(av, "AVAILABLE (+on_stock)", "phase3_available_intervals_v2.csv")
    score(un, "UNKNOWN", "phase3_unknown_intervals_v2.csv")
    print("\nreal-time guarantee (from backtest_cqr_fix.py): coverage 81.6–95.5% at every "
          "quarterly origin 2024-03..2025-09, never below the 80% promise.")


if __name__ == "__main__":
    main()
