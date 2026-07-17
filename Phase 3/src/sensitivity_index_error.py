#!/usr/bin/env python3
"""Phase 3 — sensitivity of the headline verdict to late-index error.

The 2025–26 months of the market index rest on 20–50 rows/month, so their level is
low-confidence — and the whole "AVAILABLE is ~61% fair" verdict rides on it. This script asks:
if the late index is wrong by ±10% / ±20%, does the verdict survive?

Mechanics: the CQR band scales linearly with the scoring-side index (band is computed in
reference space, then multiplied by I(t)/I(ref)), so a late-index error of factor f simply
multiplies P10/P90 by f for listings dated >= 2025-01. No refit needed — we rescale the
published interval files and recount the flags. (Training-side deflation of the 2025 rows
also shifts, but that's 561 of 10,097 rows — second-order, ignored here.)

Output: data/phase3_sensitivity_index.csv + printed table.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
FACTORS = [0.80, 0.90, 1.00, 1.10, 1.20]
LATE = "2025-01"
FLAGS = ["UNDERPRICED", "fair", "OVERPRICED", "STRONGLY_OVER"]


def reflag(df, f):
    late = (pd.to_datetime(df["last_modified_date"], errors="coerce")
            .dt.to_period("M") >= pd.Period(LATE, "M")) & df["time_adjusted"]
    s = np.where(late, f, 1.0)
    p10, p90, ask = df["P10_B"].values * s, df["P90_B"].values * s, df["asking_B"].values
    flag = np.where(ask < p10, "UNDERPRICED",
           np.where(ask > p90 * 1.3, "STRONGLY_OVER",
           np.where(ask > p90, "OVERPRICED", "fair")))
    vc = pd.Series(flag).value_counts(normalize=True) * 100
    return {k: round(float(vc.get(k, 0)), 1) for k in FLAGS} | {"n_late": int(late.sum())}


def main():
    rows = []
    for tag, fn in [("AVAILABLE", "phase3_available_intervals_v2.csv"),
                    ("UNKNOWN", "phase3_unknown_intervals_v2.csv")]:
        df = pd.read_csv(DATA / fn)
        for f in FACTORS:
            rows.append({"set": tag, "index_x": f} | reflag(df, f))
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "phase3_sensitivity_index.csv", index=False)
    for tag, g in out.groupby("set", sort=False):
        print(f"\n=== {tag} — flag shares (%) vs late-index (>= {LATE}) error "
              f"(affects {g['n_late'].iloc[0]:,} dated-2025+ rows) ===")
        print(g.drop(columns=["set", "n_late"]).to_string(index=False))
    print("\nsaved: phase3_sensitivity_index.csv")


if __name__ == "__main__":
    main()
