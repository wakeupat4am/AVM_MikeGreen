#!/usr/bin/env python3
"""Phase 3 — prediction intervals via Conformalized Quantile Regression (CQR), time-aware.

Turns the point clearing-price model into a calibrated band [P10, P50, P90] with *validated*
coverage, then flags a listing by where its asking falls relative to the band (not an arbitrary
±10% around a point).

Method
------
1. Time-adjust as in the point model: deflate training prices to a reference month, predict in
   ref space, re-inflate to the listing's date.
2. Fit LightGBM quantile models (alpha = 0.10 / 0.50 / 0.90) on the deflated log price (SOLD ×5).
3. CQR calibration (in log space, so multiplicative in price) — per district, with a global
   fallback for thin districts: widen [q_lo, q_hi] by Q so the 80% interval really covers ~80%.
4. Report empirical coverage + median interval width overall and per district.
5. Flag AVAILABLE + UNKNOWN:  asking < P10 -> UNDERPRICED ; in-band -> fair ;
   asking > P90 -> OVERPRICED ; asking > P90 by >30% -> STRONGLY_OVER.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
from build_phase3_modeling import load_splits, NUM, CAT, FEATS, TARGET
from train_phase3_clearing_time import build_index, index_for

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
RS = 42
LO_Q, HI_Q = 0.10, 0.90          # base quantile models (P10 / P90)
COV = 0.80                        # target central coverage -> CQR calibrates the band to hit ~80%
MIN_CAL = 50                      # min calibration rows for a per-district Q (else global)
BASE = dict(n_estimators=700, learning_rate=0.03, num_leaves=63, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8, random_state=RS, verbosity=-1)


def make_q(alpha):
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUM),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), CAT)])
    return Pipeline([("pre", pre), ("m", LGBMRegressor(objective="quantile", alpha=alpha, **BASE))])


def group_Q(dist, E):
    """CQR conformity quantile per district (fallback to global for thin districts)."""
    n = len(E)
    gq = {}
    glob = np.quantile(E, min(1.0, COV * (1 + 1 / n)))
    for d, idx in pd.Series(range(n)).groupby(dist.values).groups.items():
        e = E[np.array(idx)]
        gq[d] = np.quantile(e, min(1.0, COV * (1 + 1 / len(e)))) if len(e) >= MIN_CAL else glob
    return gq, glob


def main():
    sm, ref = build_index()
    train, av, un = load_splits()
    idx_tr = index_for(train["last_modified_date"], sm, ref)
    ly = np.log1p(train[TARGET].values * ref / idx_tr)          # deflated log price (ref date)
    ya = train[TARGET].values

    # ---- OOF quantile predictions (leakage-safe) in ref-log space ----
    oof = {a: np.zeros(len(train)) for a in (0.10, 0.50, 0.90)}
    for tri, vai in GroupKFold(5).split(train, groups=train["property_id"]):
        for a in oof:
            m = make_q(a); m.fit(train.iloc[tri][FEATS], ly[tri], m__sample_weight=train.iloc[tri]["w"].values)
            oof[a][vai] = m.predict(train.iloc[vai][FEATS])

    # ---- CQR calibration in log space (per district) ----
    E = np.maximum(oof[0.10] - ly, ly - oof[0.90])
    gq, glob = group_Q(train["district_name"], E)
    Qv = train["district_name"].map(lambda d: gq.get(d, glob)).values

    # OOF interval in actual-price space (re-inflate to each row's own date)
    scale = idx_tr / ref
    def band(l10, l50, l90, q, sc):
        return (np.expm1(l10 - q) * sc, np.expm1(l50) * sc, np.expm1(l90 + q) * sc)
    lo_raw = np.expm1(oof[0.10]) * scale; hi_raw = np.expm1(oof[0.90]) * scale
    lo, med, hi = band(oof[0.10], oof[0.50], oof[0.90], Qv, scale)

    cov_raw = ((ya >= lo_raw) & (ya <= hi_raw)).mean()
    cov = ((ya >= lo) & (ya <= hi)).mean()
    width = np.median(hi / np.maximum(lo, 1))
    print(f"=== INTERVAL VALIDATION (out-of-fold, target coverage {COV:.0%}) ===")
    print(f"  raw quantile coverage : {cov_raw:5.1%}")
    print(f"  CQR-calibrated cover. : {cov:5.1%}   | median P90/P10 width x{width:.2f}")
    print(f"  point (P50) MedAPE    : {np.median(np.abs(med-ya)/ya)*100:5.1f}%")

    # per-district coverage report
    dd = pd.DataFrame({"district": train["district_name"].values, "in": (ya >= lo) & (ya <= hi),
                       "w": hi / np.maximum(lo, 1)})
    rep = dd.groupby("district").agg(n=("in", "size"), coverage=("in", "mean"),
                                     med_width=("w", "median")).query("n>=100").sort_values("coverage")
    rep["coverage"] = (rep["coverage"] * 100).round(1); rep["med_width"] = rep["med_width"].round(2)
    print("\n=== per-district coverage (n>=100, target 80%) ===")
    print(rep.to_string())
    rep.to_csv(DATA / "phase3_interval_coverage.csv")

    # ---- final quantile models on all train, then score with calibrated bands ----
    models = {a: make_q(a).fit(train[FEATS], ly, m__sample_weight=train["w"].values) for a in (0.10, 0.50, 0.90)}

    def score(df, tag, fn):
        sc = index_for(df["last_modified_date"], sm, ref) / ref
        q = df["district_name"].map(lambda d: gq.get(d, glob)).fillna(glob).values
        l10 = models[0.10].predict(df[FEATS]); l50 = models[0.50].predict(df[FEATS]); l90 = models[0.90].predict(df[FEATS])
        p10, p50, p90 = band(l10, l50, l90, q, sc)
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
        print(f"\n=== SCORE (interval flags): {tag}  (n={len(df):,}) -> {fn} ===")
        for k in ["UNDERPRICED", "fair", "OVERPRICED", "STRONGLY_OVER"]:
            n = int(vc.get(k, 0)); print(f"    {k:14s} {n:6,d}  {n/len(df)*100:5.1f}%")

    score(av, "AVAILABLE (+on_stock)", "phase3_available_intervals.csv")
    score(un, "UNKNOWN", "phase3_unknown_intervals.csv")


if __name__ == "__main__":
    main()
