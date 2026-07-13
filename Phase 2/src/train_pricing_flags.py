#!/usr/bin/env python3
"""Phase 2 — per-group P90 pricing flags on AVAILABLE, for both L1 and L2 point models.

Fair range = [point x group_P10_ratio, point x group_P90_ratio], where the group ratios are the
out-of-fold P10/P90 of (actual / prediction) per district group (so the band is wider where the
model is genuinely uncertain, e.g. Hoàn Kiếm). A listing is over-priced only if its asking exceeds
the group P90 ceiling, and 'strongly over' if it exceeds it by >30%.
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
from lightgbm import LGBMRegressor
from train_final_k4 import K4, NUM, CAT, W_SOLD

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
RS = 42
BASE = dict(n_estimators=700, learning_rate=0.03, num_leaves=63, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8, random_state=RS, verbosity=-1)
F = NUM + CAT


def make(obj):
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUM),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), CAT)])
    return Pipeline([("pre", pre), ("mm", LGBMRegressor(objective=obj, **BASE))])


def run(obj, tag, sr, test):
    # --- out-of-fold predictions on SOLD+REVOKED for honest per-group ratios ---
    oof = np.zeros(len(sr))
    for tri, vai in KFold(5, shuffle=True, random_state=RS).split(sr):
        m = make(obj); m.fit(sr.iloc[tri][F], np.log1p(sr.iloc[tri].y), mm__sample_weight=sr.iloc[tri].w.values)
        oof[vai] = np.expm1(m.predict(sr.iloc[vai][F]))
    ratio = pd.Series(sr.y.values / oof, index=sr.index)
    grp = sr.dgroup.values
    r = pd.DataFrame({"g": grp, "ratio": ratio.values})
    p10 = r.groupby("g").ratio.quantile(0.10)
    p90 = r.groupby("g").ratio.quantile(0.90)
    print(f"[{tag}] per-group value band (ratio to point): "
          + " | ".join(f"G{g}:[{p10[g]:.2f},{p90[g]:.2f}]" for g in sorted(p10.index)))

    # --- final model on all SOLD+REVOKED, predict AVAILABLE ---
    m = make(obj); m.fit(sr[F], np.log1p(sr.y), mm__sample_weight=sr.w.values)
    pred = np.expm1(m.predict(test[F]))
    gt = test.dgroup.values
    floor = pred * test.dgroup.map(p10).values
    ceil = pred * test.dgroup.map(p90).values
    ask = test.y.values

    over_pt = (ask - pred) / pred * 100        # vs point
    over_cl = (ask - ceil) / ceil * 100        # vs group P90 ceiling
    flag = np.where(ask < floor, "UNDERPRICED",
           np.where(over_cl > 30, "STRONGLY_OVER",
           np.where(ask > ceil, "OVERPRICED", "fair")))
    print(f"        vs POINT:  over>30% = {(over_pt>30).mean()*100:.0f}%")
    print(f"        vs group-P90 CEILING:  over-ceiling = {(ask>ceil).mean()*100:.0f}%  |  "
          f"STRONGLY over(>30% above ceiling) = {(over_cl>30).mean()*100:.0f}%  |  fair-or-under = {(ask<=ceil).mean()*100:.0f}%")

    out = test[["property_id", "district", "dgroup", "ward_name", "total_area"]].copy()
    out["asking_B"] = (ask / 1e9).round(3)
    out["point_value_B"] = (pred / 1e9).round(3)
    out["fair_floor_P10_B"] = (floor / 1e9).round(3)
    out["fair_ceiling_P90_B"] = (ceil / 1e9).round(3)
    out["over_ceiling_pct"] = over_cl.round(1)
    out["flag"] = flag
    fn = DATA / f"available_flags_{tag}.csv"
    out.to_csv(fn, index=False)
    print(f"        saved {fn.name}  | flags: " + str(pd.Series(flag).value_counts().to_dict()))
    return out


def main():
    df = pd.read_parquet(DATA / "transaction_cleaned.parquet"); df = df[~df.suspicious_price].copy()
    df["district"] = df["district_name"].str.replace(r"^(Q\.|Quận|H\.|Huyện)\s*", "", regex=True).str.strip()
    df["dgroup"] = df["district"].map(K4).astype("Int64").astype(str)
    df["y"] = df["reference_selling_price"].astype(float); df = df[(df.y > 0) & (df.total_area > 0)]
    sr = df[df.status.isin(["SOLD", "REVOKED"])].copy(); sr["w"] = np.where(sr.status == "SOLD", W_SOLD, 1.0)
    test = df[df.available_for_sale_status == "AVAILABLE"].copy()
    print(f"train SOLD+REVOKED={len(sr)} | AVAILABLE test={len(test)}\n")
    run("regression_l1", "L1", sr, test)
    print()
    run("regression", "L2", sr, test)


if __name__ == "__main__":
    main()
