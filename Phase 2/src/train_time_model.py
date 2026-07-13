#!/usr/bin/env python3
"""Phase 2 — add normalized time features (from last_modified_date) and retrain.

- Keeps re-listings (does NOT dedup by property_id) so the model can see the same property at
  different times/prices -> it can learn the time/price-change effect.
- Validation uses GroupKFold on property_id so a house never appears in both train and validation
  (avoids the leakage the user flagged).
- Compares baseline vs +time features, and scores the AVAILABLE set (leakage-safe: AVAILABLE
  property_ids removed from training).
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
from train_final_k4 import K4, NUM, CAT, W_SOLD

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
RS = 42
CLIP = {"construction_year": (1900, 2026), "number_of_floors": (1, 15), "facade_width": (0.5, 30),
        "min_waist_width_to_main_road": (0, 50), "max_waist_width_to_main_road": (0, 50),
        "main_road_dist": (0, 5000)}
TIME = ["lm_year", "lm_month", "lm_tindex"]   # normalized time from last_modified
LGBM = dict(objective="regression", n_estimators=700, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=RS, verbosity=-1)


def load_with_relistings():
    df = pd.read_parquet(DATA / "transaction_data_26_07_10.parquet")  # RAW -> keeps re-listings
    df["district"] = df["district_name"].str.replace(r"^(Q\.|Quận|H\.|Huyện)\s*", "", regex=True).str.strip()
    df["dgroup"] = df["district"].map(K4).astype("Int64").astype(str)
    df["y"] = df["reference_selling_price"].astype(float)
    df = df[(df.y > 0) & (df.total_area > 10)].copy()
    for c, (lo, hi) in CLIP.items():
        df.loc[~df[c].between(lo, hi) & df[c].notna(), c] = np.nan
    df["has_geo"] = df["latitude"].between(20.5, 21.5) & df["longitude"].between(105.3, 106.2)
    df.loc[~df.has_geo, ["latitude", "longitude"]] = np.nan
    df["ppm2"] = df.reference_selling_price / df.total_area
    df["suspicious"] = (df.ppm2 < 20e6) | (df.ppm2 > 800e6)
    # normalized time features from last_modified_date
    lm = pd.to_datetime(df["last_modified_date"], errors="coerce")
    df["lm_year"] = lm.dt.year
    df["lm_month"] = lm.dt.month
    df["lm_tindex"] = (df.lm_year - 2023) * 12 + (df.lm_month - 1)   # months since Jan-2023
    df["listing_year"] = pd.to_datetime(df["created_date"], format="%m/%d/%Y", errors="coerce").dt.year
    return df


def pipe(num):
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), CAT)])
    return Pipeline([("pre", pre), ("mm", LGBMRegressor(**LGBM))]), num + CAT


def grouped_cv(pool, num, tag):
    gkf = GroupKFold(5)
    scores, hit = [], []
    for tri, vai in gkf.split(pool, groups=pool["property_id"]):
        tr, va = pool.iloc[tri], pool.iloc[vai]
        m, f = pipe(num)
        m.fit(tr[f], np.log1p(tr.y), mm__sample_weight=tr.w.values)
        p = np.expm1(m.predict(va[f])); a = np.abs(p - va.y.values) / va.y.values
        scores.append(np.median(a) * 100); hit.append((a <= .2).mean() * 100)
    print(f"[{tag}] grouped-CV (by property_id): MedAPE {np.mean(scores):.2f}±{np.std(scores):.2f}  Hit20 {np.mean(hit):.1f}")
    return np.mean(scores)


def main():
    df = load_with_relistings()
    df = df[~df.suspicious]
    sr = df[df.status.isin(["SOLD", "REVOKED"])].copy()
    sr["w"] = np.where(sr.status == "SOLD", W_SOLD, 1.0)
    print(f"SOLD+REVOKED rows (with re-listings): {len(sr)} | unique properties: {sr.property_id.nunique()}")
    print(f"(re-listing rows removed by dedup would have been {len(sr) - sr.property_id.nunique()})\n")

    print("=== grouped-CV: does the time feature help? (no property leaks across folds) ===")
    grouped_cv(sr, NUM, "baseline (no time)")
    grouped_cv(sr, NUM + TIME, "+ time (last_modified yr/mo/index)")

    # ---- AVAILABLE test, leakage-safe: drop training rows sharing a property with AVAILABLE ----
    av = df[df.available_for_sale_status == "AVAILABLE"].copy()
    av = av.sort_values("lm_tindex").drop_duplicates("property_id", keep="last")   # one row per available property
    sr_clean = sr[~sr.property_id.isin(av.property_id)]
    m, f = pipe(NUM + TIME); m.fit(sr_clean[f], np.log1p(sr_clean.y), mm__sample_weight=sr_clean.w.values)
    pred = np.expm1(m.predict(av[f])); ask = av.y.values; r = (ask - pred) / pred * 100
    print(f"\n=== AVAILABLE test (+time, leakage-safe n={len(av)}) ===")
    print(f"  MedAPE(vs asking) {np.median(np.abs(pred-ask)/ask)*100:.1f}%  MAE {np.mean(np.abs(pred-ask))/1e9:.2f}B  median over-ask {np.median(r):+.0f}%")
    print(f"  over>10%={ (r>10).mean()*100:.0f}%  over>30%={ (r>30).mean()*100:.0f}%")

    # feature importance of time terms
    imp = pd.Series(m.named_steps["mm"].feature_importances_, index=f)
    print(f"  time-feature importance share: {imp[TIME].sum()/imp.sum()*100:.1f}%  ({dict(imp[TIME].round(0).astype(int))})")

    # save the time-aware AVAILABLE predictions (final deliverable)
    out = av[["property_id", "district", "dgroup", "ward_name", "total_area", "lm_year", "lm_month"]].copy()
    out["asking_B"] = (ask / 1e9).round(2)
    out["time_aware_value_B"] = (pred / 1e9).round(2)
    out["over_ask_pct"] = r.round(1)
    out["flag"] = np.where(r < -10, "UNDERPRICED", np.where(r > 30, "STRONGLY_OVER",
                  np.where(r > 10, "OVERPRICED", "fair")))
    out.to_csv(DATA / "available_time_aware_predictions.csv", index=False)
    print("  saved data/available_time_aware_predictions.csv")


if __name__ == "__main__":
    main()
