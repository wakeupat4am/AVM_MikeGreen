#!/usr/bin/env python3
"""Phase 3 — lean clearing-price model: train on SOLD+REVOKED, score AVAILABLE + UNKNOWN.

- Lean common features (area, floors, facade, district, ward) so UNKNOWN is scored only on
  columns it actually has (Phase 2 showed geo adds no significant gain).
- log target, SOLD up-weighted, L1 (median-calibrated) LightGBM.
- Leakage-safe validation: GroupKFold by property_id; out-of-fold metrics overall and on SOLD only.
- Scoring -> predicted market-clearing price + mispricing flag vs the asking price, per set.
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

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
RS = 42
LGBM = dict(objective="regression_l1", n_estimators=700, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=RS, verbosity=-1)


def make():
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUM),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), CAT)])
    return Pipeline([("pre", pre), ("m", LGBMRegressor(**LGBM))])


def metrics(y, p, tag):
    y = np.asarray(y, float); p = np.asarray(p, float); ape = np.abs(p - y) / y
    return {"set": tag, "n": len(y), "MAE_B": round(np.mean(np.abs(p - y)) / 1e9, 3),
            "MAPE": round(ape.mean() * 100, 2), "MedAPE": round(np.median(ape) * 100, 2),
            "Hit20": round((ape <= .2).mean() * 100, 1)}


def flags(ask, pred):
    r = (ask - pred) / pred * 100.0
    fl = np.where(r < -10, "UNDERPRICED",
         np.where(r > 30, "STRONGLY_OVER",
         np.where(r > 10, "OVERPRICED", "fair")))
    return r, fl


def score(model, df, tag, fn):
    pred = np.expm1(model.predict(df[FEATS]))
    ask = df[TARGET].values
    r, fl = flags(ask, pred)
    out = df[["property_id", "dataset_source", "district_name", "ward_name",
              "total_area", "number_of_floors"]].copy()
    out["asking_B"] = (ask / 1e9).round(2)
    out["pred_clearing_B"] = (pred / 1e9).round(2)
    out["over_ask_pct"] = r.round(1)
    out["flag"] = fl
    out.to_csv(DATA / fn, index=False)
    vc = pd.Series(fl).value_counts()
    print(f"\n=== SCORE: {tag}  (n={len(df):,}) -> {fn} ===")
    print(f"  median asking vs clearing: {np.median(r):+.0f}%")
    for k in ["UNDERPRICED", "fair", "OVERPRICED", "STRONGLY_OVER"]:
        n = int(vc.get(k, 0))
        print(f"    {k:14s} {n:6,d}  {n / len(df) * 100:5.1f}%")


def main():
    train, av, un = load_splits()
    y = np.log1p(train[TARGET].values)
    ya = train[TARGET].values

    # ---- leakage-safe out-of-fold validation (GroupKFold by property_id) ----
    oof = np.zeros(len(train))
    for tri, vai in GroupKFold(5).split(train, groups=train["property_id"]):
        m = make(); m.fit(train.iloc[tri][FEATS], y[tri], m__sample_weight=train.iloc[tri]["w"].values)
        oof[vai] = np.expm1(m.predict(train.iloc[vai][FEATS]))
    sold = (train["status2"] == "SOLD").values
    print("=== VALIDATION — out-of-fold, leakage-safe (GroupKFold by property_id) ===")
    print(pd.DataFrame([metrics(ya, oof, "all SOLD+REVOKED"),
                        metrics(ya[sold], oof[sold], "SOLD only (truth-ish)")]).to_string(index=False))

    # ---- final fit on all train, then score the two inventory sets ----
    model = make(); model.fit(train[FEATS], y, m__sample_weight=train["w"].values)
    score(model, av, "AVAILABLE (+on_stock) — full-feature, HIGHER trust",
          "phase3_available_predictions.csv")
    score(model, un, "UNKNOWN — feature-poor + noisy asking, LOWER trust (cross-check comparables)",
          "phase3_unknown_predictions.csv")


if __name__ == "__main__":
    main()
