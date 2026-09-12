#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate deployment-oriented top-k targeting metrics on temporal test months."""
from __future__ import annotations
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PREDICTIONS = RESULTS / "test_predictions.csv"
MODELS = {
    "Logistic_RFM": "prob_Logistic_RFM",
    "Logistic_All": "prob_Logistic_All",
    "RandomForest_All": "prob_RandomForest_All",
    "LightGBM_All": "prob_LightGBM_All",
}
FRACTIONS = (0.10, 0.20, 0.30)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260827


def targeting_metrics(frame: pd.DataFrame, probability: str, fraction: float) -> dict[str, float]:
    selected_positive = 0
    selected_total = 0
    all_positive = 0
    all_total = 0
    for _, month in frame.groupby("cutoff", sort=True):
        ranked = month.sort_values(
            [probability, "CustomerID"], ascending=[False, True], kind="mergesort"
        )
        selected_n = max(1, math.ceil(len(ranked) * fraction))
        selected = ranked.iloc[:selected_n]
        selected_positive += int(selected["label"].sum())
        selected_total += len(selected)
        all_positive += int(ranked["label"].sum())
        all_total += len(ranked)
    precision = selected_positive / selected_total
    capture_rate = selected_positive / all_positive
    prevalence = all_positive / all_total
    return {
        "contact_fraction": fraction,
        "selected_n": selected_total,
        "selected_positive": selected_positive,
        "precision_at_k": precision,
        "capture_rate": capture_rate,
        "lift_at_k": precision / prevalence,
        "prevalence": prevalence,
    }


def customer_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    customers = np.sort(frame["CustomerID"].unique())
    customer_to_code = {customer: code for code, customer in enumerate(customers)}
    row_codes = frame["CustomerID"].map(customer_to_code).to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(customers), size=len(customers))
        counts = np.bincount(sampled, minlength=len(customers))
        repeated_rows = np.repeat(np.arange(len(frame)), counts[row_codes])
        boot = frame.iloc[repeated_rows]
        reference = targeting_metrics(boot, MODELS["Logistic_All"], 0.20)
        baseline = targeting_metrics(boot, MODELS["Logistic_RFM"], 0.20)
        records.append({
            "replicate": replicate,
            "precision_difference": reference["precision_at_k"] - baseline["precision_at_k"],
            "capture_rate_difference": reference["capture_rate"] - baseline["capture_rate"],
            "lift_difference": reference["lift_at_k"] - baseline["lift_at_k"],
        })
    raw = pd.DataFrame(records)
    summary = []
    for metric in ("precision_difference", "capture_rate_difference", "lift_difference"):
        values = raw[metric].to_numpy()
        summary.append({
            "reference": "Logistic_All",
            "comparator": "Logistic_RFM",
            "contact_fraction": 0.20,
            "metric": metric,
            "mean_difference": values.mean(),
            "ci_lower": np.quantile(values, 0.025),
            "ci_upper": np.quantile(values, 0.975),
            "probability_difference_gt_0": np.mean(values > 0),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        })
    return pd.DataFrame(summary)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    frame = pd.read_csv(PREDICTIONS)
    frame["cutoff"] = frame["cutoff"].astype(str)

    rows = []
    for model, probability in MODELS.items():
        for fraction in FRACTIONS:
            row = targeting_metrics(frame, probability, fraction)
            row["model"] = model
            rows.append(row)
    metrics = pd.DataFrame(rows)[[
        "model", "contact_fraction", "selected_n", "selected_positive",
        "precision_at_k", "capture_rate", "lift_at_k", "prevalence"
    ]]
    metrics.to_csv(RESULTS / "marketing_targeting_metrics.csv", index=False)

    gain_rows = []
    grid = np.arange(0.05, 1.001, 0.05)
    for model, probability in MODELS.items():
        for fraction in grid:
            row = targeting_metrics(frame, probability, float(fraction))
            gain_rows.append({
                "model": model,
                "contact_fraction": fraction,
                "capture_rate": row["capture_rate"],
                "lift_at_k": row["lift_at_k"],
            })
    gains = pd.DataFrame(gain_rows)
    gains.to_csv(RESULTS / "cumulative_gain_curves.csv", index=False)

    bootstrap = customer_bootstrap(frame)
    bootstrap.to_csv(RESULTS / "marketing_targeting_bootstrap.csv", index=False)

    labels = {
        "Logistic_RFM": "Logistic (RFM)",
        "Logistic_All": "Logistic (all features)",
        "RandomForest_All": "Random forest",
        "LightGBM_All": "LightGBM",
    }
    colors = {
        "Logistic_RFM": "#7f7f7f",
        "Logistic_All": "#1f77b4",
        "RandomForest_All": "#ff7f0e",
        "LightGBM_All": "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for model in MODELS:
        subset = gains[gains["model"] == model]
        ax.plot(
            subset["contact_fraction"], subset["capture_rate"],
            marker="o", markersize=3, linewidth=1.8,
            color=colors[model], label=labels[model],
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.2, label="Random targeting")
    ax.set_xlabel("Contact fraction")
    ax.set_ylabel("Cumulative share of positive customer-cutoff samples")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure4_cumulative_gain.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "Figure4_cumulative_gain.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(metrics.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
