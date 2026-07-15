#!/usr/bin/env python3
"""Phase 3 — assemble modeling splits from the status policy.

TRAIN  = SOLD + REVOKED (LOCKED folded into SOLD), suspicious prices removed.
         Adds `sell_label` (SOLD/LOCKED=1, REVOKED=0) and `w` (SOLD up-weighted).
SCORE  = AVAILABLE (+ON_STOCK) and UNKNOWN, kept separate (feature completeness differs).

Uses the LEAN common feature set — features every group actually has — so UNKNOWN is never
scored on columns it lacks (no geo / construction_year / road dist / shape).
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
W_SOLD = 5.0
NUM = ["total_area", "number_of_floors", "facade_width"]
CAT = ["district_name", "ward_name"]
FEATS = NUM + CAT
TARGET = "reference_selling_price"


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT:
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df


def load_splits():
    d = pd.read_parquet(DATA / "phase3_merged_dataset.parquet")
    d["status2"] = d["status"].replace({"ON_STOCK": "AVAILABLE", "LOCKED": "SOLD"})
    d = d[(d[TARGET] > 0) & (d["total_area"] > 0)].copy()

    train = d[d["status2"].isin(["SOLD", "REVOKED"]) & (~d["suspicious_price"])].copy()
    train["sell_label"] = (train["status2"] == "SOLD").astype(int)
    train["w"] = np.where(train["status2"] == "SOLD", W_SOLD, 1.0)

    score_av = d[d["status2"] == "AVAILABLE"].copy()
    score_un = d[d["status2"] == "UNKNOWN"].copy()
    return _prep(train), _prep(score_av), _prep(score_un)


def main():
    train, av, un = load_splits()
    train.to_parquet(DATA / "phase3_train.parquet", index=False)
    av.to_parquet(DATA / "phase3_score_available.parquet", index=False)
    un.to_parquet(DATA / "phase3_score_unknown.parquet", index=False)
    print(f"TRAIN (SOLD+REVOKED, clean): {len(train):,}  "
          f"(sell=1 [SOLD/LOCKED]={int(train.sell_label.sum())}, sell=0 [REVOKED]={int((train.sell_label == 0).sum())})")
    print(f"SCORE available (+on_stock): {len(av):,}")
    print(f"SCORE unknown:               {len(un):,}")
    print("saved: phase3_train / phase3_score_available / phase3_score_unknown .parquet")


if __name__ == "__main__":
    main()
