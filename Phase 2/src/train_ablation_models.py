#!/usr/bin/env python3
"""Phase 2 — 3-model ablation to test the Phase 1 hypothesis on clean transaction data.

Groups (cleaned, non-suspicious):
  REVOKED   ~9.9K  main training pool (its price distribution ≈ SOLD)
  AVAILABLE ~0.33K pricey on-stock inventory (data-augmentation lever)
  SOLD      ~0.19K realized sales -> EXTERNAL evaluation (the truth we care about)

Design:
  - Lock an internal REVOKED test (20%), never trained on.
  - 5-fold GROUPED CV (grouped by ward -> no ward leaks across folds) for internal stability.
  - Externally evaluate every model on SOLD.

Models (each changes ONE thing vs A):
  A  train REVOKED, base features
  B  train REVOKED + AVAILABLE, base features         -> does more data help generalize?
  C  train REVOKED, base + geo-KNN local price prior  -> does fine location (Phase 1 lever) help?

Target: reference_selling_price (log1p), predicted back with expm1.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neighbors import BallTree
from lightgbm import LGBMRegressor

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
RANDOM_STATE = 42
K_GEO = 8

NUM_BASE = ["total_area", "number_of_floors", "construction_year", "facade_width", "main_road_dist",
            "min_waist_width_to_main_road", "max_waist_width_to_main_road", "latitude", "longitude",
            "listing_year", "commission_rate"]
CAT = ["district", "ward_name", "shape_name", "selling_furniture"]
LGBM = dict(objective="regression_l1", n_estimators=600, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=-1)


def make_pipeline(use_geo: bool) -> Pipeline:
    num = NUM_BASE + (["geo_prior"] if use_geo else [])
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), CAT),
    ])
    return Pipeline([("pre", pre), ("m", LGBMRegressor(**LGBM))]), num + CAT


def geo_prior(train: pd.DataFrame, query: pd.DataFrame, loo: bool, k: int = K_GEO) -> pd.Series:
    """Median price/m2 of the k nearest geo-neighbours (from TRAIN only). loo=True drops self
    when query rows are also training rows. Rows without geo get NaN (model imputes)."""
    tr = train[train["has_geo"]]
    out = pd.Series(np.nan, index=query.index, dtype=float)
    q = query[query["has_geo"]]
    if len(tr) < k + 1 or len(q) == 0:
        return out
    tree = BallTree(np.radians(tr[["latitude", "longitude"]].values), metric="haversine")
    vals = tr["ppm2"].values
    d, idx = tree.query(np.radians(q[["latitude", "longitude"]].values), k=k + (1 if loo else 0))
    st = 1 if loo else 0
    out.loc[q.index] = np.median(vals[idx[:, st:st + k]], axis=1)
    return out


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / y_true
    pe = (y_pred - y_true) / y_true
    return {"n": len(y_true), "MedAPE": round(np.median(ape) * 100, 2), "MAPE": round(ape.mean() * 100, 2),
            "Hit10": round((ape <= .1).mean() * 100, 1), "Hit20": round((ape <= .2).mean() * 100, 1),
            "MedBias": round(np.median(pe) * 100, 2)}


def cv_medape(pool: pd.DataFrame, use_geo: bool) -> tuple[float, float]:
    gkf = GroupKFold(n_splits=5)
    groups = pool["ward_name"].fillna("NA").values
    scores = []
    for tr_i, va_i in gkf.split(pool, groups=groups):
        tr, va = pool.iloc[tr_i].copy(), pool.iloc[va_i].copy()
        if use_geo:
            tr["geo_prior"] = geo_prior(tr, tr, loo=True)
            va["geo_prior"] = geo_prior(tr, va, loo=False)
        model, feats = make_pipeline(use_geo)
        model.fit(tr[feats], np.log1p(tr["y"]))
        pred = np.expm1(model.predict(va[feats]))
        scores.append(np.median(np.abs(pred - va["y"]) / va["y"]) * 100)
    return float(np.mean(scores)), float(np.std(scores))


def train_eval(pool: pd.DataFrame, evals: dict, use_geo: bool):
    tr = pool.copy()
    if use_geo:
        tr["geo_prior"] = geo_prior(tr, tr, loo=True)
    model, feats = make_pipeline(use_geo)
    model.fit(tr[feats], np.log1p(tr["y"]))
    res = {}
    for name, ev in evals.items():
        ev = ev.copy()
        if use_geo:
            ev["geo_prior"] = geo_prior(tr, ev, loo=False)  # neighbours from training pool only
        pred = np.expm1(model.predict(ev[feats]))
        res[name] = metrics(ev["y"].values, pred)
    return res


def main():
    df = pd.read_parquet(DATA / "transaction_cleaned.parquet")
    df = df[~df["suspicious_price"]].copy()
    df["district"] = df["district_name"].str.replace(r"^(Q\.|Quận|H\.|Huyện)\s*", "", regex=True).str.strip()
    df["ppm2"] = df["price_per_m2"] / 1e6
    df["y"] = df["reference_selling_price"].astype(float)
    df = df[(df["y"] > 0) & (df["total_area"] > 0)]

    REVOKED = df[df["status"] == "REVOKED"]
    AVAILABLE = df[df["available_for_sale_status"] == "AVAILABLE"]
    SOLD = df[df["status"] == "SOLD"]
    print(f"groups -> REVOKED {len(REVOKED)} | AVAILABLE {len(AVAILABLE)} | SOLD {len(SOLD)}")

    rev_tr, rev_te = train_test_split(REVOKED, test_size=0.2, random_state=RANDOM_STATE,
                                      stratify=REVOKED["district"])
    print(f"locked internal REVOKED test: {len(rev_te)} | train pool: {len(rev_tr)}\n")

    evals = {"internal REVOKED test": rev_te, "external SOLD": SOLD}
    configs = {
        "A: REVOKED (base)":            (rev_tr, False),
        "B: REVOKED+AVAILABLE (base)":  (pd.concat([rev_tr, AVAILABLE]), False),
        "C: REVOKED (base + geo-KNN)":  (rev_tr, True),
    }
    rows = []
    for name, (pool, use_geo) in configs.items():
        cvm, cvs = cv_medape(pool, use_geo)
        res = train_eval(pool, evals, use_geo)
        rows.append({"model": name, "cv_MedAPE": f"{cvm:.2f}±{cvs:.2f}",
                     "intTest_MedAPE": res["internal REVOKED test"]["MedAPE"],
                     "intTest_Hit20": res["internal REVOKED test"]["Hit20"],
                     "SOLD_MedAPE": res["external SOLD"]["MedAPE"],
                     "SOLD_Hit20": res["external SOLD"]["Hit20"],
                     "SOLD_MedBias": res["external SOLD"]["MedBias"]})
        print(f"[{name}]  CV {cvm:.2f}±{cvs:.2f}  | internal {res['internal REVOKED test']} | SOLD {res['external SOLD']}")

    out = pd.DataFrame(rows)
    out.to_csv(DATA.parent / "results_ablation.csv", index=False)
    print("\n=== ABLATION SUMMARY ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
