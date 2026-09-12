#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibration diagnostics for the deployment-time repeat-purchase models."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEED = 20260827
N_BOOT = 2000

MODEL_COLUMNS = {
    "Logistic_RFM": "prob_Logistic_RFM",
    "Logistic_All": "prob_Logistic_All",
    "RandomForest_All": "prob_RandomForest_All",
    "LightGBM_All": "prob_LightGBM_All",
}


def calibration_bins(y, probability, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(probability, edges, right=False) - 1, 0, n_bins - 1)
    rows = []
    for bin_id in range(n_bins):
        mask = ids == bin_id
        if not mask.any():
            continue
        rows.append(
            {
                "bin": bin_id + 1,
                "lower": edges[bin_id],
                "upper": edges[bin_id + 1],
                "n": int(mask.sum()),
                "mean_probability": float(probability[mask].mean()),
                "observed_rate": float(y[mask].mean()),
                "absolute_gap": float(
                    abs(probability[mask].mean() - y[mask].mean())
                ),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(bins):
    total = bins["n"].sum()
    return float((bins["n"] / total * bins["absolute_gap"]).sum())


def customer_cluster_brier_difference(frame):
    y = frame["label"].to_numpy()
    reference = frame["prob_Logistic_All"].to_numpy()
    comparator = frame["prob_Logistic_RFM"].to_numpy()
    groups = frame["CustomerID"].to_numpy()
    unique_groups = np.unique(groups)
    locations = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(N_BOOT):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([locations[group] for group in sampled])
        ref_brier = np.mean((y[index] - reference[index]) ** 2)
        cmp_brier = np.mean((y[index] - comparator[index]) ** 2)
        differences.append(ref_brier - cmp_brier)
    differences = np.asarray(differences)
    return pd.DataFrame(
        [
            {
                "reference": "Logistic_All",
                "comparator": "Logistic_RFM",
                "metric": "brier",
                "mean_difference": float(differences.mean()),
                "ci_lower": float(np.quantile(differences, 0.025)),
                "ci_upper": float(np.quantile(differences, 0.975)),
                "probability_difference_lt_0": float(np.mean(differences < 0)),
                "bootstrap_replicates": len(differences),
            }
        ]
    )


def main():
    frame = pd.read_csv(RESULTS / "test_predictions.csv")
    y = frame["label"].to_numpy()
    metrics_rows = []
    bin_frames = []
    for model, column in MODEL_COLUMNS.items():
        probability = frame[column].to_numpy()
        bins = calibration_bins(y, probability)
        bins.insert(0, "model", model)
        bin_frames.append(bins)
        metrics_rows.append(
            {
                "model": model,
                "brier": brier_score_loss(y, probability),
                "log_loss": log_loss(y, probability),
                "ece_10": expected_calibration_error(bins),
                "mean_probability": probability.mean(),
                "observed_rate": y.mean(),
            }
        )

    metrics = pd.DataFrame(metrics_rows).sort_values("brier")
    all_bins = pd.concat(bin_frames, ignore_index=True)
    bootstrap = customer_cluster_brier_difference(frame)
    metrics.to_csv(RESULTS / "calibration_metrics.csv", index=False)
    all_bins.to_csv(RESULTS / "calibration_bins.csv", index=False)
    bootstrap.to_csv(RESULTS / "calibration_bootstrap.csv", index=False)

    colors = {
        "Logistic_RFM": "#808080",
        "Logistic_All": "#1f77b4",
        "RandomForest_All": "#2ca02c",
        "LightGBM_All": "#ff7f0e",
    }
    labels = {
        "Logistic_RFM": "Logistic (RFM)",
        "Logistic_All": "Logistic (all features)",
        "RandomForest_All": "Random forest",
        "LightGBM_All": "LightGBM",
    }
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1.2, label="Perfect calibration")
    for model in MODEL_COLUMNS:
        part = all_bins[all_bins["model"] == model]
        ax.plot(
            part["mean_probability"],
            part["observed_rate"],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=colors[model],
            label=labels[model],
        )
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed repurchase rate")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "Figure3_calibration_reliability.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "Figure3_calibration_reliability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(metrics.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
