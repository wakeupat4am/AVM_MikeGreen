#!/usr/bin/env python3
"""Phase 3 — "will it sell?" classifier on the ONE real label (SOLD vs REVOKED).

Everything else in Phase 3 models *asking* prices; this is the only task with a genuine outcome
label: a listing either cleared (SOLD, +LOCKED) or was pulled without selling (REVOKED). It lifts
the product from "is this asking price fair?" to "will this listing sell?" — and **fuses the two**:
the mispricing signal from the clearing engine (asking vs its own predicted clearing band) becomes
a *feature* here, testing the natural hypothesis *less over-priced -> more likely to clear.*

Hard truths this script respects (measured, not assumed):
  - **Only 206 positives** among 10,097, base rate 2.0%. Accuracy is meaningless; we rank (AUC,
    lift) and report bootstrap CIs so we never over-claim on thin positives.
  - **Positives are front-loaded** (82% in 2023H2–2024H1) and the base rate itself drifts over time.
    So the honest test is temporal, and calendar time must not leak as a feature.

Validation
----------
1. PRIMARY — **prequential / expanding-window** (real-time): for each test month m, train the
   classifier on all rows with month < m, score month m. Every listing is scored once, by a model
   that saw only its past. Pool all out-of-time (score, label) pairs -> one AUC + bootstrap CI.
2. OPTIMISTIC baseline — random GroupKFold AUC (by property_id), to quantify (again) how much a
   random fold flatters a temporally-structured problem.
3. Ablation — LEAN vs LEAN+mispricing, so the mispricing feature's marginal value is explicit.
4. Product view — rank by P(sell); precision & lift in the top decile vs the 2.0% base rate.

Leakage notes: the mispricing feature is a GroupKFold out-of-fold clearing prediction (the clearing
model never sees a row's own price); its only residual optimism is that a row's clearing fold-mates
can include later dates — minor, and the prequential classifier split is the primary defense. The
classifier never receives calendar month, price level, or the label as a feature.

Output: data/phase3_sellability_scores.csv (+ printed metrics).
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
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from build_phase3_modeling import load_splits, NUM, CAT, FEATS, TARGET
from train_phase3_clearing import make as make_clearing
from train_phase3_clearing_time import build_index, index_for

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
RS = 42
MISF = "mispricing"                                   # asking/clearing - 1 (OOF, time-aware)
RICH_NUM = ["construction_year", "latitude", "longitude", "main_road_dist"]  # 88–100% in SOLD+REVOKED
LEAN = FEATS
FULL = FEATS + [MISF]
RICH = FEATS + RICH_NUM
TEST_MONTHS = pd.period_range("2024-01", "2026-06", freq="M")   # prequential test window
MIN_TRAIN, MIN_TRAIN_POS = 800, 25                    # need enough history (esp. positives) to fit
CLF = dict(objective="binary", n_estimators=400, learning_rate=0.03, num_leaves=31,
           min_child_samples=30, subsample=0.8, colsample_bytree=0.8, random_state=RS, verbosity=-1)


def make_clf(feats):
    cat = [c for c in feats if c in CAT]
    num = [c for c in feats if c not in CAT]          # everything non-categorical is numeric
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat)])
    return Pipeline([("pre", pre), ("m", LGBMClassifier(**CLF))])


def oof_mispricing(train, sm, ref):
    """Leakage-safe out-of-fold clearing prediction -> mispricing = asking/clearing - 1."""
    idx = index_for(train["last_modified_date"], sm, ref)
    y = np.log1p(train[TARGET].values * ref / idx)                 # deflate to ref date
    clearing = np.zeros(len(train))
    for tri, vai in GroupKFold(5).split(train, groups=train["property_id"]):
        m = make_clearing(); m.fit(train.iloc[tri][FEATS], y[tri], m__sample_weight=train.iloc[tri]["w"].values)
        clearing[vai] = np.expm1(m.predict(train.iloc[vai][FEATS])) * idx[vai] / ref
    return train[TARGET].values / np.maximum(clearing, 1) - 1.0


def boot_auc(y, s, n=2000, seed=RS):
    """Bootstrap 95% CI for AUC (resample rows; keep only resamples with both classes)."""
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            out.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def prequential(train, feats):
    """Expanding-window: train on month<m, score month m; pool out-of-time scores (once per row)."""
    score = np.full(len(train), np.nan)
    for m in TEST_MONTHS:
        te = (train["month"] == m).values
        tr = (train["month"] < m).values
        if te.sum() == 0 or tr.sum() < MIN_TRAIN or train.loc[tr, "sell_label"].sum() < MIN_TRAIN_POS:
            continue
        spw = max(1.0, (tr.sum() - train.loc[tr, "sell_label"].sum()) / max(1, train.loc[tr, "sell_label"].sum()))
        clf = make_clf(feats)
        clf.set_params(m__scale_pos_weight=spw)
        clf.fit(train.loc[tr, feats], train.loc[tr, "sell_label"].values)
        score[te] = clf.predict_proba(train.loc[te, feats])[:, 1]
    return score


def groupkfold_auc(train, feats):
    """Optimistic random-fold AUC (by property_id)."""
    oof = np.zeros(len(train))
    y = train["sell_label"].values
    for tri, vai in GroupKFold(5).split(train, groups=train["property_id"]):
        spw = (y[tri] == 0).sum() / max(1, (y[tri] == 1).sum())
        clf = make_clf(feats); clf.set_params(m__scale_pos_weight=spw)
        clf.fit(train.iloc[tri][feats], y[tri])
        oof[vai] = clf.predict_proba(train.iloc[vai][feats])[:, 1]
    return oof


def lift_table(y, s, base):
    """Top-decile precision & lift over base rate (the ranking product view)."""
    o = np.argsort(-s)
    k = max(1, int(0.10 * len(s)))
    top = y[o][:k]
    prec = top.mean()
    return {"top_decile_n": k, "top_decile_precision": round(prec * 100, 2),
            "lift_vs_base": round(prec / base, 2), "captured_positives_pct": round(top.sum() / y.sum() * 100, 1)}


def main():
    sm, ref = build_index()
    train, _, _ = load_splits()
    train["month"] = pd.to_datetime(train["last_modified_date"], errors="coerce").dt.to_period("M")
    train = train[train["month"].notna()].copy().reset_index(drop=True)
    y = train["sell_label"].values
    base = y.mean()
    print(f"label: SOLD/LOCKED={int(y.sum())} vs REVOKED={int((y==0).sum())}  base rate {base:.1%}")

    for c in RICH_NUM:
        if c in train:
            train[c] = pd.to_numeric(train[c], errors="coerce")

    train[MISF] = oof_mispricing(train, sm, ref)
    print(f"\nmispricing (asking/clearing-1) — hypothesis 'less over-priced clears more':")
    print(f"  SOLD    median {np.median(train.loc[y==1, MISF]):+.1%}   mean {train.loc[y==1, MISF].mean():+.1%}")
    print(f"  REVOKED median {np.median(train.loc[y==0, MISF]):+.1%}   mean {train.loc[y==0, MISF].mean():+.1%}")

    print("\n=== AUC — feature ablation, temporal (real-time) vs random-fold ===")
    rows = []
    pooled = {}
    for name, feats in [("lean", LEAN), ("lean+mispricing", FULL), ("rich (geo+year+road)", RICH)]:
        pq = prequential(train, feats)
        m = ~np.isnan(pq)
        auc_t = roc_auc_score(y[m], pq[m]); lo_t, hi_t = boot_auc(y[m], pq[m])
        rf = groupkfold_auc(train, feats)
        auc_r = roc_auc_score(y, rf); lo_r, hi_r = boot_auc(y, rf)
        pooled[name] = (pq, m)
        rows.append({"features": name,
                     "temporal_AUC": f"{auc_t:.3f} [{lo_t:.3f},{hi_t:.3f}]", "temporal_n": int(m.sum()),
                     "temporal_pos": int(y[m].sum()),
                     "randomfold_AUC": f"{auc_r:.3f} [{lo_r:.3f},{hi_r:.3f}]"})
    print(pd.DataFrame(rows).to_string(index=False))

    # product view on the temporal (real-time) scores of the better model
    pq, m = pooled["lean+mispricing"]
    print(f"\n=== ranking product view (real-time scores, n={int(m.sum()):,}, base {base:.1%}) ===")
    print(f"  lean+mispricing: {lift_table(y[m], pq[m], base)}")
    pql, ml = pooled["lean"]
    print(f"  lean only      : {lift_table(y[ml], pql[ml], base)}")

    out = train[["property_id", "district_name", "ward_name", "total_area", "number_of_floors",
                 "last_modified_date", "sell_label"]].copy()
    out["mispricing_pct"] = (train[MISF] * 100).round(1)
    out["p_sell_realtime"] = np.where(m, pq, np.nan).round(4)
    out.to_csv(DATA / "phase3_sellability_scores.csv", index=False)
    print("\nsaved: phase3_sellability_scores.csv")


if __name__ == "__main__":
    main()
