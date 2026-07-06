from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common import EVAL_DIR, RANDOM_STATE, load_processed_train_test, metric_row
from outlier_handling import prepare_outlier_artifacts
from train_catboost_region_subtype import (
    CATBOOST_PARAMS,
    choose_policy,
    train_family_models,
)


OUT_DIR = EVAL_DIR / "improvements" / "catboost_selected_tuning"
MAX_TRAIN_ROWS = 50000
MAX_VAL_ROWS = 12000

CANDIDATES = [
    {"name": "cb_d6_lr003_l2_3_iter_600", "params": {"iterations": 600, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 3.0, "min_data_in_leaf": 20}},
    {"name": "cb_d7_lr003_l2_5_iter_800", "params": {"iterations": 800, "learning_rate": 0.03, "depth": 7, "l2_leaf_reg": 5.0, "min_data_in_leaf": 20}},
]


def validation_metric(df: pd.DataFrame, pred_col: str) -> dict:
    metrics = metric_row(df["price_vnd"], df[pred_col])
    return {
        "Median_APE": float(metrics["Median_APE"]),
        "HitRate_20_pct": float(metrics["HitRate_20_pct"]),
        "HitRate_10_pct": float(metrics["HitRate_10_pct"]),
        "MAPE": float(metrics["MAPE"]),
    }


def build_validation_hybrid(df: pd.DataFrame, routing_policy: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lookup = routing_policy.set_index(["addr_last_1", "property_subtype"])["chosen_model"].to_dict()
    out["selected_model"] = out.apply(lambda r: lookup.get((r["addr_last_1"], r["property_subtype"]), "global"), axis=1)
    out["pred_price_vnd_hybrid"] = out["pred_price_vnd_global"]
    out.loc[out["selected_model"] == "region", "pred_price_vnd_hybrid"] = (
        out.loc[out["selected_model"] == "region", "pred_price_vnd_region"]
        .fillna(out.loc[out["selected_model"] == "region", "pred_price_vnd_global"])
    )
    out.loc[out["selected_model"] == "subtype", "pred_price_vnd_hybrid"] = (
        out.loc[out["selected_model"] == "subtype", "pred_price_vnd_subtype"]
        .fillna(out.loc[out["selected_model"] == "subtype", "pred_price_vnd_global"])
    )
    out.loc[out["selected_model"] == "region_subtype", "pred_price_vnd_hybrid"] = (
        out.loc[out["selected_model"] == "region_subtype", "pred_price_vnd_region_subtype"]
        .fillna(out.loc[out["selected_model"] == "region_subtype", "pred_price_vnd_region"])
        .fillna(out.loc[out["selected_model"] == "region_subtype", "pred_price_vnd_global"])
    )
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, _ = load_processed_train_test()
    train_split, val_split = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_df["property_subtype"],
    )
    artifacts = prepare_outlier_artifacts(train_split, {"validation": val_split})
    tr = artifacts.train_df
    va = artifacts.other_dfs["validation"]

    def sample_frame(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if len(df) <= max_rows:
            return df.copy()
        frac = max_rows / len(df)
        idx = (
            df.groupby("property_subtype", group_keys=False)
            .apply(lambda part: part.sample(max(1, int(round(len(part) * frac))), random_state=RANDOM_STATE).index)
            .explode()
            .astype(int)
            .tolist()
        )
        out = df.loc[idx].copy()
        if len(out) > max_rows:
            out = out.sample(max_rows, random_state=RANDOM_STATE)
        return out.reset_index(drop=True)

    tr = sample_frame(tr, MAX_TRAIN_ROWS)
    va = sample_frame(va, MAX_VAL_ROWS)

    rows = []
    for candidate in CANDIDATES:
        params = CATBOOST_PARAMS.copy()
        params.update(candidate["params"])
        family = train_family_models(tr, va.copy(), params)
        routing_policy = choose_policy(family["validation_predictions"], family["train_group_sizes"])
        pred_df = build_validation_hybrid(family["validation_predictions"], routing_policy)

        region_subtype_metrics = validation_metric(pred_df.loc[pred_df["pred_price_vnd_region_subtype"].notna()].copy(), "pred_price_vnd_region_subtype")
        hybrid_metrics = validation_metric(pred_df, "pred_price_vnd_hybrid")
        score = region_subtype_metrics["Median_APE"] + hybrid_metrics["Median_APE"]

        rows.append(
            {
                "candidate": candidate["name"],
                "score": score,
                "region_subtype_Median_APE": region_subtype_metrics["Median_APE"],
                "region_subtype_HitRate_20_pct": region_subtype_metrics["HitRate_20_pct"],
                "hybrid_Median_APE": hybrid_metrics["Median_APE"],
                "hybrid_HitRate_20_pct": hybrid_metrics["HitRate_20_pct"],
                "params_json": json.dumps(params, ensure_ascii=False),
            }
        )

    result = pd.DataFrame(rows).sort_values(["score", "hybrid_Median_APE", "region_subtype_Median_APE"]).reset_index(drop=True)
    result.to_csv(OUT_DIR / "tuning_results.csv", index=False)
    best = result.iloc[0]
    (OUT_DIR / "best_params.json").write_text(best["params_json"], encoding="utf-8")
    print(result.to_string(index=False))
    print("\nBEST_PARAMS_JSON")
    print(best["params_json"])


if __name__ == "__main__":
    main()
