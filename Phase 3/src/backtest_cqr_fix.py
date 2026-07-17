#!/usr/bin/env python3
"""Phase 3 — fix the CQR drift under-coverage, and prove the fix with the rolling backtest.

backtest_rolling_origin.py showed the published 80% band really covers 67–77% at most origins
under a temporal split. Candidate fixes, gridded here and judged by real-time coverage:

  cal_months  3 vs 6      longer calibration window (more scores, but staler)
  rho         1.0 vs 0.85 recency weight per month of age on conformity scores
                          (weighted conformal: recent errors count more under drift)
  nominal     0.80/0.85/0.90  calibrate the band to a higher nominal level to land at 80% real

Index: hedonic_nowcast (the backtest-winning causal variant). Selection rule: weighted-mean
coverage in [78, 88]%, maximize min-origin coverage, tie-break on narrower width.

Output: data/phase3_cqr_fix_grid.csv + recommended config.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from build_phase3_modeling import load_splits, FEATS, TARGET
from train_phase3_intervals import make_q
from backtest_rolling_origin import (load_market_rows, hedonic_index_path, idx_hedonic,
                                     ORIGINS, H, MIN_TEST)

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
GRID_CAL = [3, 6]
GRID_RHO = [1.0, 0.85]
GRID_NOM = [0.80, 0.85, 0.90]
TARGET_COV = 0.80


def wquantile(x, w, q):
    """Weighted quantile with the conformal finite-sample bump on effective n."""
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    q = min(1.0, q * (1 + 1 / n_eff))
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w[o].sum()
    return float(x[o][min(np.searchsorted(cw, q), len(x) - 1)])


def main():
    train, _, _ = load_splits()
    train["month"] = pd.to_datetime(train["last_modified_date"], errors="coerce").dt.to_period("M")
    train = train[train["month"].notna()].copy()
    hpath = hedonic_index_path(load_market_rows())

    rows = []
    for o in ORIGINS:
        T = pd.Period(o, "M")
        tr = train[train["month"] <= T]
        te = train[(train["month"] > T) & (train["month"] <= T + H)]
        if len(te) < MIN_TEST:
            continue
        tr_m, te_m = tr["month"].values, te["month"].values
        defl = np.array([idx_hedonic(hpath, T, T) / idx_hedonic(hpath, m, T) for m in tr_m])
        infl = np.array([idx_hedonic(hpath, m, m) / idx_hedonic(hpath, T, m) for m in te_m])
        ly = np.log1p(tr[TARGET].values * defl)
        ya = te[TARGET].values

        for cal in GRID_CAL:
            fit_m = (tr["month"] < T - cal + 1).values
            cal_m = ~fit_m
            if fit_m.sum() < 300 or cal_m.sum() < 40:
                continue
            qm = {a: make_q(a).fit(tr[fit_m][FEATS], ly[fit_m],
                                   m__sample_weight=tr[fit_m]["w"].values) for a in (0.10, 0.90)}
            l10c = qm[0.10].predict(tr[cal_m][FEATS]); l90c = qm[0.90].predict(tr[cal_m][FEATS])
            E = np.maximum(l10c - ly[cal_m], ly[cal_m] - l90c)
            age = np.array([(T - m).n for m in tr_m[cal_m]], float)     # months before T
            l10 = qm[0.10].predict(te[FEATS]); l90 = qm[0.90].predict(te[FEATS])
            for rho in GRID_RHO:
                w = rho ** age
                for nom in GRID_NOM:
                    Q = wquantile(E, w, nom)
                    lo = np.expm1(l10 - Q) * infl
                    hi = np.expm1(l90 + Q) * infl
                    rows.append({"origin": o, "n_test": len(te), "cal_months": cal,
                                 "rho": rho, "nominal": nom,
                                 "coverage": ((ya >= lo) & (ya <= hi)).mean(),
                                 "width": float(np.median(hi / np.maximum(lo, 1)))})

    g = pd.DataFrame(rows)
    g.to_csv(DATA / "phase3_cqr_fix_grid.csv", index=False)

    agg = (g.groupby(["cal_months", "rho", "nominal"])
             .apply(lambda d: pd.Series({
                 "cov_wmean": np.average(d["coverage"], weights=d["n_test"]),
                 "cov_min": d["coverage"].min(),
                 "width_med": d["width"].median(),
                 "n_origins": len(d)}), include_groups=False)
             .reset_index())
    agg["cov_wmean"] = (agg["cov_wmean"] * 100).round(1)
    agg["cov_min"] = (agg["cov_min"] * 100).round(1)
    agg["width_med"] = agg["width_med"].round(2)
    print("=== CQR fix grid — real-time coverage across origins (target 80%) ===")
    print(agg.sort_values(["cov_min", "cov_wmean"], ascending=False).to_string(index=False))

    ok = agg[(agg["cov_wmean"] >= 78) & (agg["cov_wmean"] <= 88)]
    if len(ok):
        best = ok.sort_values(["cov_min", "width_med"], ascending=[False, True]).iloc[0]
        print(f"\nRECOMMENDED: cal_months={int(best.cal_months)}, rho={best.rho}, "
              f"nominal={best.nominal} -> wmean {best.cov_wmean}%, min {best.cov_min}%, "
              f"width x{best.width_med}")
        print("\nper-origin coverage for the recommended config:")
        sel = g[(g.cal_months == best.cal_months) & (g.rho == best.rho) & (g.nominal == best.nominal)]
        print(sel.assign(coverage=(sel.coverage * 100).round(1), width=sel.width.round(2))
                 [["origin", "n_test", "coverage", "width"]].to_string(index=False))
    print("\nsaved: phase3_cqr_fix_grid.csv")


if __name__ == "__main__":
    main()
