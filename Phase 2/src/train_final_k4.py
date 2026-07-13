#!/usr/bin/env python3
"""Phase 2 — final model with K=4 district groups.

Train/validation: SOLD + REVOKED (SOLD up-weighted vs REVOKED).
Test: AVAILABLE (the unsold inventory) -> used to detect over/under-pricing vs the
model's market-clearing value.

Reports: validation accuracy, and on the AVAILABLE test set the share of listings that are
under- / fairly / over-priced, plus the most common under-price magnitude.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
RS = 42
W_SOLD = 5.0          # SOLD rows weighted 5x vs REVOKED
FAIR_BAND = 10.0      # +-10% around the model value counts as "fairly priced"

K4 = {
    "Đông Anh": 1, "Gia Lâm": 1, "Hoài Đức": 1, "Long Biên": 1, "Hoàng Mai": 1, "Bắc Từ Liêm": 1, "Hà Đông": 1,
    "Thanh Trì": 2, "Nam Từ Liêm": 2, "Hai Bà Trưng": 2, "Thanh Xuân": 2, "Tây Hồ": 2,
    "Đống Đa": 3, "Ba Đình": 3, "Cầu Giấy": 3,
    "Hoàn Kiếm": 4,
}
NUM = ["total_area", "number_of_floors", "construction_year", "facade_width", "main_road_dist",
       "min_waist_width_to_main_road", "max_waist_width_to_main_road", "latitude", "longitude",
       "listing_year", "commission_rate"]
CAT = ["district", "ward_name", "shape_name", "selling_furniture", "dgroup"]
LGBM = dict(objective="regression_l1", n_estimators=700, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=RS, verbosity=-1)


def metrics(y, p, tag):
    y = np.asarray(y, float); p = np.asarray(p, float); ape = np.abs(p - y) / y
    return {"set": tag, "n": len(y), "MAE_B": round(np.mean(np.abs(p - y)) / 1e9, 3),
            "MedAE_B": round(np.median(np.abs(p - y)) / 1e9, 3), "MAPE": round(ape.mean() * 100, 2),
            "MedAPE": round(np.median(ape) * 100, 2), "Hit20": round((ape <= .2).mean() * 100, 1),
            "MedBias": round(np.median((p - y) / y) * 100, 2)}


def main():
    df = pd.read_parquet(DATA / "transaction_cleaned.parquet"); df = df[~df.suspicious_price].copy()
    df["district"] = df["district_name"].str.replace(r"^(Q\.|Quận|H\.|Huyện)\s*", "", regex=True).str.strip()
    df["dgroup"] = df["district"].map(K4).astype("Int64").astype(str)
    df["y"] = df["reference_selling_price"].astype(float)
    df = df[(df.y > 0) & (df.total_area > 0)]

    trainval = df[df.status.isin(["SOLD", "REVOKED"])].copy()
    test = df[df.available_for_sale_status == "AVAILABLE"].copy()
    trainval["w"] = np.where(trainval.status == "SOLD", W_SOLD, 1.0)

    tr, va = train_test_split(trainval, test_size=0.2, random_state=RS, stratify=trainval.district)
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUM),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), CAT)])
    model = Pipeline([("pre", pre), ("m", LGBMRegressor(**LGBM))])
    feats = NUM + CAT
    model.fit(tr[feats], np.log1p(tr.y), m__sample_weight=tr.w.values)

    print("train:", len(tr), "(SOLD", int((tr.status=='SOLD').sum()), "w=%g) | val:"%W_SOLD, len(va),
          "| test AVAILABLE:", len(test), "\n")

    # ---- validation accuracy (real market) ----
    pv = np.expm1(model.predict(va[feats]))
    print("=== VALIDATION metrics (SOLD+REVOKED held-out) ===")
    print(pd.DataFrame([metrics(va.y, pv, "validation (all)"),
                        metrics(va[va.status=="SOLD"].y, np.expm1(model.predict(va[va.status=="SOLD"][feats])), "validation (SOLD only)")]).to_string(index=False))

    # ---- TEST = AVAILABLE: over/under-pricing vs model market value ----
    pred = np.expm1(model.predict(test[feats]))
    ask = test.y.values
    resid = (ask - pred) / pred * 100.0            # + => asking above model value (OVERpriced)
    under = resid < -FAIR_BAND                       # asking well below model value
    over = resid > FAIR_BAND
    fair = ~under & ~over
    print("\n=== TEST: AVAILABLE listings vs model market value (n=%d) ===" % len(test))
    print(f"  UNDER-priced (asking < value by >{FAIR_BAND:.0f}%): {under.mean()*100:5.1f}%")
    print(f"  fairly priced (within +-{FAIR_BAND:.0f}%):           {fair.mean()*100:5.1f}%")
    print(f"  OVER-priced (asking > value by >{FAIR_BAND:.0f}%):  {over.mean()*100:5.1f}%")
    print(f"  median asking-vs-value gap: {np.median(resid):+.1f}%")

    # most common under-price magnitude (modal discount bucket among underpriced)
    disc = -resid[under]                             # how far below value (positive %)
    bins = np.arange(10, 90, 10)
    h, edges = np.histogram(disc, bins=list(bins) + [200])
    mi = int(np.argmax(h))
    lo_e, hi_e = edges[mi], edges[mi + 1]
    print(f"\n  among UNDER-priced listings ({under.sum()} of {len(test)}):")
    print(f"    most common under-price band: {lo_e:.0f}-{hi_e:.0f}% below value  ({h[mi]} listings, {h[mi]/max(under.sum(),1)*100:.0f}% of underpriced)")
    print(f"    median under-price: {np.median(disc):.0f}% below value")

    out = test[["property_id", "district", "dgroup", "ward_name", "total_area", "price_bil"]].copy()
    out["model_value_B"] = (pred / 1e9).round(2)
    out["asking_vs_value_pct"] = resid.round(1)
    out["flag"] = np.where(under, "UNDERPRICED", np.where(over, "OVERPRICED", "fair"))
    out.to_csv(DATA / "available_pricing_flags.csv", index=False)
    print("\nsaved data/available_pricing_flags.csv")


if __name__ == "__main__":
    main()
