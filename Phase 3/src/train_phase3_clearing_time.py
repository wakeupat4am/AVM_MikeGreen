#!/usr/bin/env python3
"""Phase 3 — TIME-AWARE clearing-price model (deterministic market-index adjustment).

The Hà Nội market rose ~2.4x across 2023-2026, so a model trained on the 2023-24-heavy
transaction pool under-values later listings and overstates their over-pricing. Fix (transparent,
non-extrapolating):

  1. Build a monthly market index I(t) = median price/m² across all dated members (smoothed).
  2. DEFLATE every training price to a reference month:  price_ref = price * I(ref)/I(t).
     Train the lean model on price_ref  ->  it predicts the clearing price *at the reference date*.
  3. RE-INFLATE each prediction to the listing's own date:  clearing(t) = pred_ref * I(t)/I(ref),
     then compare to the asking price (also at date t).  Dates beyond the data are clamped
     (no extrapolation).  Undated UNKNOWN rows fall back to the reference level (flagged).

Outputs time-adjusted predictions + a no-time vs time comparison (the key temporal insight).
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from build_phase3_modeling import load_splits, FEATS, TARGET
from train_phase3_clearing import make, metrics, flags

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
REF_MONTH = "2024-06"     # reference date the model predicts at (cancels out of the over-ask ratio)


def build_index(ref_month=REF_MONTH):
    d = pd.read_parquet(DATA / "phase3_merged_dataset.parquet")
    d["lm"] = pd.to_datetime(d["last_modified_date"], errors="coerce")
    d = d[d["lm"].notna() & (~d["suspicious_price"]) & (d["price_per_m2"] > 0)]
    m = d.groupby(d["lm"].dt.to_period("M"))["price_per_m2"].median().sort_index()
    sm = m.rolling(3, center=True, min_periods=1).median()          # smooth thin months
    ref = float(sm.get(pd.Period(ref_month, "M"), sm.median()))
    return sm, ref


def index_for(dates, sm, ref):
    per = pd.to_datetime(dates, errors="coerce").dt.to_period("M")
    first, last = sm.index.min(), sm.index.max()
    out = np.empty(len(per))
    for i, p in enumerate(per):
        if pd.isna(p):
            out[i] = ref                                            # undated -> reference level
        else:
            p = min(max(p, first), last)                            # clamp (no extrapolation)
            out[i] = sm.loc[p]
    return out


def score(model, df, sm, ref, tag, fn):
    idx = index_for(df["last_modified_date"], sm, ref)
    p_ref = np.expm1(model.predict(df[FEATS]))
    pred = p_ref * idx / ref                                        # re-inflate to listing date
    ask = df[TARGET].values
    r, fl = flags(ask, pred)
    dated = pd.to_datetime(df["last_modified_date"], errors="coerce").notna().values
    out = df[["property_id", "dataset_source", "district_name", "ward_name",
              "total_area", "number_of_floors", "last_modified_date"]].copy()
    out["asking_B"] = (ask / 1e9).round(2)
    out["pred_clearing_B"] = (pred / 1e9).round(2)
    out["over_ask_pct"] = r.round(1)
    out["flag"] = fl
    out["time_adjusted"] = dated
    out.to_csv(DATA / fn, index=False)
    vc = pd.Series(fl).value_counts()
    print(f"\n=== SCORE (time-aware): {tag}  (n={len(df):,}, dated={dated.sum():,}) -> {fn} ===")
    print(f"  median asking vs clearing: {np.median(r):+.0f}%")
    for k in ["UNDERPRICED", "fair", "OVERPRICED", "STRONGLY_OVER"]:
        n = int(vc.get(k, 0)); print(f"    {k:14s} {n:6,d}  {n/len(df)*100:5.1f}%")
    return np.median(r)


def main():
    sm, ref = build_index()
    print(f"market index: {sm.index.min()}..{sm.index.max()}  ref={REF_MONTH} ({ref/1e6:.0f}M/m²)  "
          f"range {sm.min()/1e6:.0f}->{sm.max()/1e6:.0f}M/m² (+{(sm.max()/sm.min()-1)*100:.0f}%)")

    train, av, un = load_splits()
    idx_tr = index_for(train["last_modified_date"], sm, ref)
    price_ref = train[TARGET].values * ref / idx_tr                 # deflate to reference date
    y = np.log1p(price_ref)
    ya = train[TARGET].values

    # ---- leakage-safe OOF validation on ACTUAL prices (predict at ref, re-inflate to own date) ----
    oof = np.zeros(len(train))
    for tri, vai in GroupKFold(5).split(train, groups=train["property_id"]):
        m = make(); m.fit(train.iloc[tri][FEATS], y[tri], m__sample_weight=train.iloc[tri]["w"].values)
        p_ref = np.expm1(m.predict(train.iloc[vai][FEATS]))
        oof[vai] = p_ref * idx_tr[vai] / ref
    sold = (train["status2"] == "SOLD").values
    print("\n=== VALIDATION — time-aware, out-of-fold (GroupKFold by property_id) ===")
    print(pd.DataFrame([metrics(ya, oof, "all SOLD+REVOKED"),
                        metrics(ya[sold], oof[sold], "SOLD only (truth-ish)")]).to_string(index=False))

    model = make(); model.fit(train[FEATS], y, m__sample_weight=train["w"].values)
    score(model, av, sm, ref, "AVAILABLE (+on_stock)", "phase3_available_predictions_timeadj.csv")
    score(model, un, sm, ref, "UNKNOWN", "phase3_unknown_predictions_timeadj.csv")


if __name__ == "__main__":
    main()
