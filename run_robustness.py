#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robustness analysis for leakage-safe Online Retail prediction."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from lightgbm import LGBMClassifier

from run_experiment import load_data, build_snapshot, metrics, best_threshold, SEED

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
N_BOOT = 2000


def paired_cluster_bootstrap(y, p_ref, p_cmp, pred_ref, pred_cmp, groups, seed):
    """Paired bootstrap by customer, retaining repeated monthly observations."""
    y = np.asarray(y)
    p_ref = np.asarray(p_ref)
    p_cmp = np.asarray(p_cmp)
    pred_ref = np.asarray(pred_ref)
    pred_cmp = np.asarray(pred_cmp)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    diffs = {"auc": [], "pr_auc": [], "f1": []}

    for _ in range(N_BOOT):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_indices[group] for group in sampled_groups])
        y_boot = y[idx]
        if np.unique(y_boot).size < 2:
            continue
        diffs["auc"].append(
            roc_auc_score(y_boot, p_ref[idx]) - roc_auc_score(y_boot, p_cmp[idx])
        )
        diffs["pr_auc"].append(
            average_precision_score(y_boot, p_ref[idx])
            - average_precision_score(y_boot, p_cmp[idx])
        )
        diffs["f1"].append(
            f1_score(y_boot, pred_ref[idx], zero_division=0)
            - f1_score(y_boot, pred_cmp[idx], zero_division=0)
        )

    rows = []
    for metric_name, values in diffs.items():
        values = np.asarray(values)
        rows.append(
            {
                "metric": metric_name,
                "mean_difference": values.mean(),
                "ci_lower": np.quantile(values, 0.025),
                "ci_upper": np.quantile(values, 0.975),
                "probability_difference_gt_0": np.mean(values > 0),
                "bootstrap_replicates": len(values),
            }
        )
    return rows


def main():
    data = load_data()
    cutoffs = pd.date_range("2011-04-01", "2011-11-01", freq="MS")
    panel = pd.concat([build_snapshot(data, cutoff) for cutoff in cutoffs], ignore_index=True)
    train = panel[panel.cutoff <= "2011-08-01"].copy()
    valid = panel[panel.cutoff == "2011-09-01"].copy()
    test = panel[panel.cutoff >= "2011-10-01"].copy()

    feature_all = [c for c in panel.columns if c not in ["CustomerID", "label", "cutoff"]]
    feature_rfm = ["recency", "frequency", "monetary"]
    specs = {
        "Logistic_RFM": (
            feature_rfm,
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
        ),
        "Logistic_All": (
            feature_all,
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
        ),
        "RandomForest_All": (
            feature_all,
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED,
            ),
        ),
        "LightGBM_All": (
            feature_all,
            LGBMClassifier(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=-1,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
                verbosity=-1,
            ),
        ),
    }

    outputs = {}
    prediction_table = test[["CustomerID", "cutoff", "label"]].reset_index(drop=True)
    for model_name, (features, model) in specs.items():
        model.fit(train[features], train.label)
        valid_probability = model.predict_proba(valid[features])[:, 1]
        threshold = best_threshold(valid.label.to_numpy(), valid_probability)
        test_probability = model.predict_proba(test[features])[:, 1]
        test_prediction = (test_probability >= threshold).astype(int)
        outputs[model_name] = {
            "probability": test_probability,
            "prediction": test_prediction,
            "threshold": threshold,
        }
        prediction_table[f"prob_{model_name}"] = test_probability
        prediction_table[f"pred_{model_name}"] = test_prediction
    prediction_table.to_csv(OUT / "test_predictions.csv", index=False)

    monthly_rows = []
    test_cutoff = test.cutoff.reset_index(drop=True)
    test_label = test.label.reset_index(drop=True)
    for model_name, output in outputs.items():
        for cutoff in sorted(test_cutoff.unique()):
            mask = (test_cutoff == cutoff).to_numpy()
            monthly_rows.append(
                {
                    "model": model_name,
                    "cutoff": pd.Timestamp(cutoff).strftime("%Y-%m-%d"),
                    "n": int(mask.sum()),
                    "positive_rate": test_label[mask].mean(),
                    **metrics(
                        test_label[mask].to_numpy(),
                        output["probability"][mask],
                        output["threshold"],
                    ),
                }
            )
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(OUT / "monthly_metrics.csv", index=False)

    reference_name = "Logistic_All"
    reference = outputs[reference_name]
    bootstrap_rows = []
    for index, comparator_name in enumerate(
        ["Logistic_RFM", "RandomForest_All", "LightGBM_All"]
    ):
        comparator = outputs[comparator_name]
        comparison_rows = paired_cluster_bootstrap(
            test.label.to_numpy(),
            reference["probability"],
            comparator["probability"],
            reference["prediction"],
            comparator["prediction"],
            test.CustomerID.to_numpy(),
            SEED + index,
        )
        for row in comparison_rows:
            row["reference"] = reference_name
            row["comparator"] = comparator_name
            bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(OUT / "paired_bootstrap.csv", index=False)

    print("Monthly metrics")
    print(monthly.to_string(index=False))
    print("\nPaired customer-cluster bootstrap: Logistic_All minus comparator")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
