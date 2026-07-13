#!/usr/bin/env python3
"""Phase 2 — robustness (paired bootstrap) + full out-of-fold geo-feature model.

(1) Bootstrap the SOLD MedAPE difference between models to see if improvements exclude 0.
(2) Model D: rich geo features (all OUT-OF-FOLD to avoid leakage):
      local_median_ppsqm, local_p10_ppsqm, local_p90_ppsqm,
      neighbor_count, neighbor_price_iqr, nearest_neighbor_distance
    with distance-weighted quantiles, radius->KNN fallback in sparse areas,
    swept over K in {5,8,12,20} x radius in {200,250,300} m.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import BallTree
from lightgbm import LGBMRegressor

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
RS = 42
EARTH = 6371000.0
NUM_BASE = ["total_area", "number_of_floors", "construction_year", "facade_width", "main_road_dist",
            "min_waist_width_to_main_road", "max_waist_width_to_main_road", "latitude", "longitude",
            "listing_year", "commission_rate"]
CAT = ["district", "ward_name", "shape_name", "selling_furniture"]
RICH = ["local_median_ppm2", "local_p10_ppm2", "local_p90_ppm2", "neighbor_count",
        "neighbor_iqr", "nn_dist"]
LGBM = dict(objective="regression_l1", n_estimators=600, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=RS, verbosity=-1)


def pipe(num_extra):
    num = NUM_BASE + num_extra
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), CAT)])
    return Pipeline([("pre", pre), ("m", LGBMRegressor(**LGBM))]), num + CAT


def wquantile(v, q, w):
    if len(v) == 0:
        return np.nan
    o = np.argsort(v); v = v[o]; w = w[o]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return float(np.interp(q, cw, v))


def geo_features(src, qry, k, radius_m, min_nb=5, weighted=True, loo=False):
    res = pd.DataFrame(np.nan, index=qry.index, columns=RICH)
    s = src[src["has_geo"]]; q = qry[qry["has_geo"]]
    if len(s) < k + 1 or len(q) == 0:
        return res
    tree = BallTree(np.radians(s[["latitude", "longitude"]].values), metric="haversine")
    vals = s["ppm2"].values
    qr = np.radians(q[["latitude", "longitude"]].values)
    ind, dst = tree.query_radius(qr, r=radius_m / EARTH, return_distance=True, sort_results=True)
    kd, kidx = tree.query(qr, k=k + (1 if loo else 0))
    out = np.full((len(q), 6), np.nan)
    for i in range(len(q)):
        idx, d = ind[i], dst[i]
        if loo and len(idx) and d[0] < 1e-9:
            idx, d = idx[1:], d[1:]
        if len(idx) < min_nb:                      # radius -> KNN fallback in sparse areas
            st = 1 if loo else 0
            idx, d = kidx[i][st:st + k], kd[i][st:st + k]
        if len(idx) == 0:
            continue
        v = vals[idx]; dm = d * EARTH
        if weighted:
            w = 1.0 / (dm + 50.0)
            med, p10, p90 = wquantile(v, .5, w), wquantile(v, .1, w), wquantile(v, .9, w)
        else:
            med, p10, p90 = np.median(v), np.percentile(v, 10), np.percentile(v, 90)
        iqr = (np.percentile(v, 75) - np.percentile(v, 25)) if len(v) >= 4 else np.nan
        out[i] = [med, p10, p90, len(v), iqr, dm[0] if len(dm) else np.nan]
    res.loc[q.index] = out
    return res


def oof_geo(train, cfg):
    kf = KFold(5, shuffle=True, random_state=RS)
    parts = []
    for tri, vai in kf.split(train):
        parts.append(geo_features(train.iloc[tri], train.iloc[vai], loo=False, **cfg))
    return pd.concat(parts).reindex(train.index)


def simple_prior(src, qry, loo, k=8):
    s = src[src["has_geo"]]; out = pd.Series(np.nan, index=qry.index, dtype=float)
    q = qry[qry["has_geo"]]
    if len(s) < k + 1 or len(q) == 0:
        return out
    tree = BallTree(np.radians(s[["latitude", "longitude"]].values), metric="haversine")
    v = s["ppm2"].values
    _, idx = tree.query(np.radians(q[["latitude", "longitude"]].values), k=k + (1 if loo else 0))
    st = 1 if loo else 0
    out.loc[q.index] = np.median(v[idx[:, st:st + k]], axis=1)
    return out


def ape(y, p):
    return np.abs(np.asarray(p, float) - np.asarray(y, float)) / np.asarray(y, float)


def fit_predict(train, evals, num_extra):
    model, feats = pipe(num_extra)
    model.fit(train[feats], np.log1p(train["y"]))
    return {name: np.expm1(model.predict(ev[feats])) for name, ev in evals.items()}


def paired_bootstrap(ape_a, ape_x, B=5000):
    n = len(ape_a); rng = np.random.default_rng(0)
    diffs = np.empty(B)
    for b in range(B):
        ix = rng.integers(0, n, n)
        diffs[b] = np.median(ape_a[ix]) * 100 - np.median(ape_x[ix]) * 100  # +ve => X better
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return diffs.mean(), lo, hi, (diffs > 0).mean() * 100


def main():
    df = pd.read_parquet(DATA / "transaction_cleaned.parquet")
    df = df[~df["suspicious_price"]].copy()
    df["district"] = df["district_name"].str.replace(r"^(Q\.|Quận|H\.|Huyện)\s*", "", regex=True).str.strip()
    df["ppm2"] = df["price_per_m2"] / 1e6
    df["y"] = df["reference_selling_price"].astype(float)
    df = df[(df["y"] > 0) & (df["total_area"] > 0)]
    REV, SOLD = df[df.status == "REVOKED"], df[df.status == "SOLD"]
    rev_tr, rev_te = train_test_split(REV, test_size=0.2, random_state=RS, stratify=REV["district"])
    evals = {"int": rev_te, "SOLD": SOLD}
    print(f"train {len(rev_tr)} | internal test {len(rev_te)} | SOLD {len(SOLD)}\n")

    # ---- Model A (base) ----
    predA = fit_predict(rev_tr, evals, [])
    apeA = ape(SOLD["y"].values, predA["SOLD"])

    # ---- Model C (simple geo prior, k=8, out-of-fold via LOO) ----
    tC = rev_tr.copy(); tC["geo_prior"] = simple_prior(tC, tC, loo=True)
    eC = {n: e.assign(geo_prior=simple_prior(tC, e, loo=False)) for n, e in evals.items()}
    predC = fit_predict(tC, eC, ["geo_prior"])
    apeC = ape(SOLD["y"].values, predC["SOLD"])

    # ---- Model D: rich OOF geo features, sweep K x radius ----
    print("=== Model D sweep (rich OOF geo features): internal & SOLD MedAPE ===")
    sweep = []
    best = None
    for k in (5, 8, 12, 20):
        for r in (200, 250, 300):
            cfg = dict(k=k, radius_m=r, weighted=True)
            tD = rev_tr.copy()
            tD[RICH] = oof_geo(tD, cfg)
            eD = {n: e.join(geo_features(tD, e, loo=False, **cfg)) for n, e in evals.items()}
            predD = fit_predict(tD, eD, RICH)
            im = np.median(ape(rev_te["y"].values, predD["int"])) * 100
            sm = np.median(ape(SOLD["y"].values, predD["SOLD"])) * 100
            sweep.append({"k": k, "radius": r, "int_MedAPE": round(im, 2), "SOLD_MedAPE": round(sm, 2)})
            if best is None or im < best["im"]:
                best = {"cfg": cfg, "im": im, "sm": sm, "apeD": ape(SOLD["y"].values, predD["SOLD"])}
    print(pd.DataFrame(sweep).to_string(index=False))
    print(f"\nbest-by-internal config: {best['cfg']}  internal {best['im']:.2f}  SOLD {best['sm']:.2f}")
    apeD = best["apeD"]

    # ---- SOLD point metrics ----
    print("\n=== SOLD MedAPE (point) ===")
    print(f"A {np.median(apeA)*100:.2f} | C {np.median(apeC)*100:.2f} | D {np.median(apeD)*100:.2f}")

    # ---- Paired bootstrap on SOLD (5000) ----
    print("\n=== Paired bootstrap of SOLD MedAPE difference (positive = better than A), 95% CI ===")
    for label, apx in [("C - A (simple geo prior)", apeC), ("D - A (rich OOF geo)", apeD)]:
        m, lo, hi, pbetter = paired_bootstrap(apeA, apx)
        sig = "SIGNIFICANT (CI excludes 0)" if lo > 0 else "not significant (CI includes 0)"
        print(f"{label:28s} improvement {m:+.2f} pts  95%CI [{lo:+.2f}, {hi:+.2f}]  P(better)={pbetter:.0f}%  -> {sig}")


if __name__ == "__main__":
    main()
