"""Score a single listing with the final production AVM and print a serving decision.

Output:
    Prediction: 5.2B
    Interval:   4.6B - 5.9B
    Confidence: Tier A
    Decision:   AUTO_VALUATION

Usage:
    python3 score_listing.py --json '{"area_m2":56,"num_floors":3,"property_subtype":"normal_house",
                                       "addr_last_1":"hồ chí minh","addr_last_2":"quận 1",
                                       "asking_price_vnd":5000000000}'
    python3 score_listing.py --input my_listing.json
    python3 score_listing.py --template          # print a blank input template
    python3 score_listing.py --demo              # score a few real test listings (actual vs predicted)

Provide whatever fields you have; missing features are imputed by the model. `asking_price_vnd`
is optional but recommended — it powers the out-of-domain / implausible-price guardrails.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # silence sklearn feature-name notices for clean serving output

import joblib
import numpy as np
import pandas as pd

from common import EVAL_DIR, is_commercial_listing

ARTIFACT = EVAL_DIR.parent / "models" / "final_scorer.joblib"

# Fields a user can provide. Everything else the model imputes.
BASE_NUM = ["area_m2", "num_floors", "frontage_m", "road_width_m", "width_m", "length_m", "bedrooms", "bathrooms"]
TEXT_FLAGS = ["is_frontage_text", "is_alley_text", "is_business_text", "is_car_access_text", "is_corner_text",
              "is_villa_text", "has_legal_text", "is_new_house_text", "is_renovated_text", "is_project_text",
              "is_urban_area_text"]
CATS = ["property_subtype", "addr_last_1", "addr_last_2", "addr_last_3", "street_proxy", "legal", "door_direction",
        "source", "project_name"]


def fmt_b(v: float) -> str:
    return f"{v / 1e9:.1f}B"


def build_feature_row(d: dict) -> pd.DataFrame:
    row = {}
    for k in BASE_NUM:
        row[k] = d.get(k, np.nan)
    for k in TEXT_FLAGS:
        row[k] = d.get(k, 0)
    row["text_len"] = d.get("text_len", 0)
    row["word_count"] = d.get("word_count", 0)
    row["listing_year"] = d.get("listing_year", 2025)
    row["listing_month"] = d.get("listing_month", 6)
    a, f, fr, rw = row["area_m2"], row["num_floors"], row["frontage_m"], row["road_width_m"]
    row["floor_area_m2"] = (a * f) if pd.notna(a) and pd.notna(f) else np.nan
    row["floors_x_area"] = (f * a) if pd.notna(a) and pd.notna(f) else np.nan
    row["frontage_x_road_width"] = (fr * rw) if pd.notna(fr) and pd.notna(rw) else np.nan
    row["floors_x_frontage"] = (f * fr) if pd.notna(f) and pd.notna(fr) else np.nan
    row["floors_x_business"] = (f * row["is_business_text"]) if pd.notna(f) else np.nan
    row["car_access_x_road_width"] = (row["is_car_access_text"] * rw) if pd.notna(rw) else np.nan
    row["area_x_frontage"] = (a * fr) if pd.notna(a) and pd.notna(fr) else np.nan
    for k in CATS:
        row[k] = d.get(k, "missing")
    # for the commercial gate
    row["all_text_lower"] = str(d.get("title", "") or "") + " " + str(d.get("description", "") or "")
    return pd.DataFrame([row])


def _predict(model_cols, X):
    model, cols = model_cols
    return float(np.expm1(model.predict(X[cols]))[0])


def score(d: dict, art: dict) -> dict:
    X = build_feature_row(d)
    region = d.get("addr_last_1", "missing")
    subtype = d.get("property_subtype", "missing")
    k = art["shrink_k"]

    # Shrinkage blend: global -> region -> region_subtype, weighted n/(n+k).
    log_est = np.log1p(_predict(art["global"], X))
    if region in art["region"]:
        n = art["region_sizes"].get(region, 0)
        a = n / (n + k)
        log_est = a * np.log1p(max(_predict(art["region"][region], X), 0)) + (1 - a) * log_est
    if (region, subtype) in art["region_subtype"]:
        n = art["region_subtype_sizes"].get((region, subtype), 0)
        a = n / (n + k)
        log_est = a * np.log1p(max(_predict(art["region_subtype"][(region, subtype)], X), 0)) + (1 - a) * log_est
    point = float(np.expm1(log_est))

    # Prediction interval: the global quantile models give a band around the *global* median;
    # rescale that relative band onto the region-aware point so the point always sits inside.
    global_median = _predict(art["global"], X)
    p10, p90 = _predict(art["p10"], X), _predict(art["p90"], X)
    gm = max(global_median, 1.0)
    rel_lo, rel_hi = min(p10, p90) / gm, max(p10, p90) / gm
    lo, hi = point * rel_lo, point * rel_hi
    rel_width = rel_hi - rel_lo

    # ---- Guardrails (out-of-scope / implausible -> REVIEW) ----
    reasons = []
    if bool(is_commercial_listing(X).iloc[0]):
        reasons.append("commercial/hotel (out of scope)")
    text = (str(d.get("title", "") or "") + " " + str(d.get("description", "") or "")).lower()
    if re.search(r"căn hộ|can ho|chung cư|chung cu|apartment|condotel|penthouse|duplex", text):
        reasons.append("apartment/condo (out of scope — AVM values landed houses only)")
    asking = d.get("asking_price_vnd")
    area = d.get("area_m2")
    if asking and area:
        ppm2 = asking / area
        if ppm2 < art["min_ppm2"]:
            reasons.append(f"price/m2 {ppm2/1e6:.0f}M below {art['min_ppm2']/1e6:.0f}M floor")
    if asking and not (lo <= asking <= hi):
        reasons.append("asking price outside P10-P90 band")

    # ---- Confidence tier from interval width ----
    if rel_width <= art["tier_q40"]:
        tier = "A"
    elif rel_width <= art["tier_q80"]:
        tier = "B"
    else:
        tier = "C"

    if reasons:
        decision = "REVIEW"
    elif tier == "A":
        decision = "AUTO_VALUATION"
    elif tier == "B":
        decision = "VALUATION_WITH_RANGE"
    else:
        decision = "REVIEW"
        reasons.append("low confidence (wide interval)")

    return {"point": point, "lo": lo, "hi": hi, "tier": tier, "decision": decision, "reasons": reasons}


def render(res: dict) -> str:
    lines = [f"Prediction: {fmt_b(res['point'])}",
             f"Interval:   {fmt_b(res['lo'])} - {fmt_b(res['hi'])}",
             f"Confidence: Tier {res['tier']}",
             f"Decision:   {res['decision']}"]
    if res["reasons"]:
        lines.append(f"Reason:     {'; '.join(res['reasons'])}")
    return "\n".join(lines)


TEMPLATE = {
    "area_m2": None, "num_floors": None, "frontage_m": None, "road_width_m": None,
    "bedrooms": None, "bathrooms": None,
    "property_subtype": "normal_house | frontage_house | alley_house | alley_frontage_house | villa | townhouse_project | shophouse",
    "addr_last_1": "province (e.g. hồ chí minh)", "addr_last_2": "district (e.g. quận 1)",
    "asking_price_vnd": "optional, powers the plausibility guardrails",
    "title": "optional text", "description": "optional text",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, help="listing as an inline JSON string")
    ap.add_argument("--input", type=str, help="path to a JSON file with the listing")
    ap.add_argument("--template", action="store_true", help="print a blank input template")
    ap.add_argument("--demo", action="store_true", help="score real test listings (actual vs predicted)")
    args = ap.parse_args()

    if args.template:
        print(json.dumps(TEMPLATE, ensure_ascii=False, indent=2))
        return
    if not ARTIFACT.exists():
        raise SystemExit(f"Model artifact not found: {ARTIFACT}\nRun: python3 build_final_scorer.py")
    art = joblib.load(ARTIFACT)

    if args.demo:
        from common import PROCESSED_DATA_DIR
        te = pd.read_parquet(PROCESSED_DATA_DIR / "test_dataset.parquet")
        samp = te[(te["price_per_m2"] >= 20e6) & (te["area_m2"].between(40, 200))].sample(5, random_state=7)
        for _, r in samp.iterrows():
            d = {c: r[c] for c in BASE_NUM + CATS + TEXT_FLAGS if c in r and pd.notna(r[c])}
            d["asking_price_vnd"] = float(r["price_vnd"])
            res = score(d, art)
            print(f"[{r['property_subtype']} | {r['addr_last_1']} | {r['area_m2']:.0f}m2]")
            print(render(res))
            print(f"  ACTUAL:   {fmt_b(float(r['price_vnd']))}   (error {abs(res['point']-r['price_vnd'])/r['price_vnd']*100:.1f}%)\n")
        return

    if args.json:
        d = json.loads(args.json)
    elif args.input:
        d = json.loads(Path(args.input).read_text(encoding="utf-8"))
    else:
        raise SystemExit("Provide --json '{...}', --input file.json, --demo, or --template")
    print(render(score(d, art)))


if __name__ == "__main__":
    main()
