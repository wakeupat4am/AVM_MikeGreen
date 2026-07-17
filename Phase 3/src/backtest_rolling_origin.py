#!/usr/bin/env python3
"""Phase 3 — rolling-origin (walk-forward) backtest: the defensibility test.

Every published Phase 3 number was validated with random GroupKFold, and the market index was
smoothed with a *centered* window over the full timeline — both let the model see the future.
This script asks the only question an auditor asks: **would it have worked in real time?**

Design: stand at origin month T, use ONLY data <= T (model + index), predict the SOLD+REVOKED
listings of months T+1..T+3 (their asking prices), slide T quarterly. Five index variants isolate
each error source:

  no_time         no index at all (the un-timed model)
  leaky           centered 3-mo smoothing over ALL months — the current README method (upper
                  bound; quantifies how much the published 18% benefited from look-ahead)
  causal_frozen   one-sided smoothing, index frozen at T, carried forward into the test window
                  ("refresh the model quarterly, never update the index")
  causal_nowcast  one-sided smoothing, index updated through each listing's own month
                  ("index refreshes monthly as data arrives" — the realistic production mode)
  hedonic_nowcast index = month coefficients of log(price/m2) ~ months + district + lean feats,
                  re-estimated at each listing's month (composition-controlled nowcast)

Note: a listing's own calendar month is included when building its nowcast index value (a
within-month simplification; the index is monthly, so days can't be split).

Also backtests CQR interval coverage under a *temporal* calibration split (calib = last 3 months
<= T), since conformal's 80% guarantee assumes exchangeability that a time split violates.

Output: data/phase3_backtest_results.csv (+ per-origin printout).
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from build_phase3_modeling import load_splits, NUM, CAT, FEATS, TARGET
from train_phase3_clearing import make, metrics
from train_phase3_intervals import make_q

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ORIGINS = ["2024-03", "2024-06", "2024-09", "2024-12", "2025-03", "2025-06", "2025-09"]
H = 3                 # test horizon: months T+1..T+H
MIN_TEST = 30
COV = 0.80            # CQR target coverage
CAL_MONTHS = 3        # temporal calibration window (last months <= T)


# ---------------------------------------------------------------- index construction
def load_market_rows():
    """All dated, price-sane rows of the merged dataset (same pool as build_index)."""
    d = pd.read_parquet(DATA / "phase3_merged_dataset.parquet")
    d["lm"] = pd.to_datetime(d["last_modified_date"], errors="coerce")
    d = d[d["lm"].notna() & (~d["suspicious_price"]) & (d["price_per_m2"] > 0)].copy()
    d["month"] = d["lm"].dt.to_period("M")
    return d


def median_indices(mkt):
    """Monthly median price/m2 -> leaky (centered) and causal (right-aligned) smoothed series."""
    m = mkt.groupby("month")["price_per_m2"].median().sort_index()
    leaky = m.rolling(3, center=True, min_periods=1).median()
    causal = m.rolling(3, min_periods=1).median()        # right-aligned: months <= t only
    return leaky, causal


def hedonic_index_path(mkt):
    """For each cutoff month c: fit log(price/m2) ~ month + district + lean feats on rows <= c,
    return {cutoff -> {month -> exp(beta_month)}} (unnormalised; ratios are what matter)."""
    df = mkt.copy()
    df["log_ppm"] = np.log(df["price_per_m2"])
    df["log_area"] = np.log(df["total_area"].clip(lower=1))
    for c in ["number_of_floors", "facade_width"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    months = sorted(df["month"].unique())
    dist = pd.get_dummies(df["district_name"].fillna("NA"), prefix="d", drop_first=True)
    mon = pd.get_dummies(df["month"].astype(str), prefix="m")           # no drop -> read all betas
    X_all = np.column_stack([mon.values, dist.values,
                             df[["log_area", "number_of_floors", "facade_width"]].values])
    y_all = df["log_ppm"].values
    mcols = list(mon.columns)
    path = {}
    for c in months:
        mask = (df["month"] <= c).values
        Xc, yc = X_all[mask], y_all[mask]
        keep = Xc.std(axis=0) > 0                                        # drop empty future-month dummies
        beta = np.zeros(X_all.shape[1])
        beta[keep] = np.linalg.lstsq(Xc[:, keep], yc, rcond=None)[0]
        path[c] = {pd.Period(col[2:], "M"): float(np.exp(beta[i]))
                   for i, col in enumerate(mcols) if keep[i]}
    return path


def idx_median(series, month, cutoff):
    """Causal median-index value for `month`, knowing data only <= cutoff (clamp both ends)."""
    known = series[series.index <= cutoff]
    m = min(max(month, known.index.min()), known.index.max())
    return float(known.loc[m])


def idx_hedonic(path, month, cutoff):
    fit = path[max(k for k in path if k <= cutoff)]
    m = min(max(month, min(fit)), max(fit))
    while m not in fit and m > min(fit):                                 # skip gap months
        m -= 1
    return fit[m]


# ---------------------------------------------------------------- backtest core
def run_variant(name, tr, te, deflate_tr, infl_te):
    """Fit the lean L1 model on deflated train prices, re-inflate test predictions, score."""
    y = np.log1p(tr[TARGET].values * deflate_tr)
    m = make()
    m.fit(tr[FEATS], y, m__sample_weight=tr["w"].values)
    pred = np.expm1(m.predict(te[FEATS])) * infl_te
    return metrics(te[TARGET].values, pred, name)


def cqr_coverage(tr, te, defl_tr, infl_te, T):
    """Temporal-split CQR: fit quantiles on months < T-CAL, calibrate on the last CAL months,
    measure band coverage on the test window."""
    cal_start = T - CAL_MONTHS + 1
    fit_m = tr["month"] < cal_start
    cal_m = ~fit_m
    if fit_m.sum() < 300 or cal_m.sum() < 40:
        return None
    ly = np.log1p(tr[TARGET].values * defl_tr)
    qm = {}
    for a in (0.10, 0.50, 0.90):
        qm[a] = make_q(a)
        qm[a].fit(tr[fit_m][FEATS], ly[fit_m], m__sample_weight=tr[fit_m]["w"].values)
    l10c = qm[0.10].predict(tr[cal_m][FEATS]); l90c = qm[0.90].predict(tr[cal_m][FEATS])
    E = np.maximum(l10c - ly[cal_m], ly[cal_m] - l90c)
    Q = np.quantile(E, min(1.0, COV * (1 + 1 / len(E))))
    l10 = qm[0.10].predict(te[FEATS]); l90 = qm[0.90].predict(te[FEATS])
    lo = np.expm1(l10 - Q) * infl_te
    hi = np.expm1(l90 + Q) * infl_te
    ya = te[TARGET].values
    return {"cov_raw": ((ya >= np.expm1(l10) * infl_te) & (ya <= np.expm1(l90) * infl_te)).mean(),
            "cov_cqr": ((ya >= lo) & (ya <= hi)).mean(),
            "width": float(np.median(hi / np.maximum(lo, 1))), "n_cal": int(cal_m.sum())}


def main():
    train, _, _ = load_splits()
    train["month"] = pd.to_datetime(train["last_modified_date"], errors="coerce").dt.to_period("M")
    train = train[train["month"].notna()].copy()

    mkt = load_market_rows()
    leaky, causal = median_indices(mkt)
    hpath = hedonic_index_path(mkt)
    print(f"market pool: {len(mkt):,} dated rows, {leaky.index.min()}..{leaky.index.max()}")

    rows, cov_rows = [], []
    for o in ORIGINS:
        T = pd.Period(o, "M")
        tr = train[train["month"] <= T]
        te = train[(train["month"] > T) & (train["month"] <= T + H)]
        if len(te) < MIN_TEST:
            print(f"\n--- origin {o}: test n={len(te)} < {MIN_TEST}, skipped")
            continue
        sold_te = int((te["status2"] == "SOLD").sum())
        print(f"\n--- origin {o}: train n={len(tr):,}  test n={len(te):,} "
              f"({T+1}..{T+H}, SOLD={sold_te})")

        tr_m, te_m = tr["month"].values, te["month"].values
        # per-variant (deflate_train, inflate_test) factors; ref month = T (cancels in ratios)
        iT_c = idx_median(causal, T, T)
        variants = {
            "no_time":        (np.ones(len(tr)), np.ones(len(te))),
            "leaky":          (np.array([float(leaky.loc[T]) / float(leaky.loc[m]) for m in tr_m]),
                               np.array([float(leaky.loc[m]) / float(leaky.loc[T]) for m in te_m])),
            "causal_frozen":  (np.array([iT_c / idx_median(causal, m, T) for m in tr_m]),
                               np.full(len(te), 1.0)),                     # index clamped at T
            "causal_nowcast": (np.array([iT_c / idx_median(causal, m, T) for m in tr_m]),
                               np.array([idx_median(causal, m, m) / iT_c for m in te_m])),
            "hedonic_nowcast": (np.array([idx_hedonic(hpath, T, T) / idx_hedonic(hpath, m, T) for m in tr_m]),
                                np.array([idx_hedonic(hpath, m, m) / idx_hedonic(hpath, T, m) for m in te_m])),
        }
        res = []
        for name, (dtr, ite) in variants.items():
            r = run_variant(name, tr, te, dtr, ite)
            r.update(origin=o, n_test=len(te), sold_test=sold_te)
            res.append(r); rows.append(r)
        print(pd.DataFrame(res)[["set", "MedAPE", "MAPE", "Hit20"]].to_string(index=False))

        cov = cqr_coverage(tr, te, variants["causal_nowcast"][0], variants["causal_nowcast"][1], T)
        if cov:
            cov.update(origin=o, n_test=len(te))
            cov_rows.append(cov)
            print(f"  CQR (temporal calib, n={cov['n_cal']}): raw {cov['cov_raw']:.1%} "
                  f"-> calibrated {cov['cov_cqr']:.1%} (target {COV:.0%}), width x{cov['width']:.2f}")

    res = pd.DataFrame(rows)
    res.to_csv(DATA / "phase3_backtest_results.csv", index=False)
    if cov_rows:
        pd.DataFrame(cov_rows).to_csv(DATA / "phase3_backtest_coverage.csv", index=False)

    print("\n=== SUMMARY — MedAPE by origin x variant (test = next 3 months, real-time) ===")
    print(res.pivot(index="origin", columns="set", values="MedAPE")
             [["no_time", "leaky", "causal_frozen", "causal_nowcast", "hedonic_nowcast"]].to_string())
    print("\n=== weighted overall (by test n) ===")
    for v, g in res.groupby("set"):
        w = g["n_test"].values
        print(f"  {v:16s} MedAPE {np.average(g['MedAPE'], weights=w):5.2f}  "
              f"Hit20 {np.average(g['Hit20'], weights=w):5.1f}")
    if cov_rows:
        c = pd.DataFrame(cov_rows)
        print(f"\n=== CQR coverage under temporal split (target {COV:.0%}) ===")
        print(c[["origin", "n_test", "n_cal", "cov_raw", "cov_cqr", "width"]]
              .assign(cov_raw=lambda d: (d.cov_raw * 100).round(1),
                      cov_cqr=lambda d: (d.cov_cqr * 100).round(1),
                      width=lambda d: d.width.round(2)).to_string(index=False))
    print("\nsaved: phase3_backtest_results.csv, phase3_backtest_coverage.csv")


if __name__ == "__main__":
    main()
