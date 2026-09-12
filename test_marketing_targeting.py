#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for snapshot construction and marketing targeting metrics."""
from pathlib import Path
import unittest

import pandas as pd

import run_marketing_targeting as targeting
import run_experiment as experiment


class MarketingTargetingTests(unittest.TestCase):
    def test_perfect_ranking_with_monthly_budget(self):
        frame = pd.DataFrame({
            "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8],
            "cutoff": ["m1"] * 4 + ["m2"] * 4,
            "label": [1, 0, 0, 0, 1, 0, 0, 0],
            "probability": [0.9, 0.4, 0.3, 0.2, 0.8, 0.5, 0.3, 0.1],
        })
        result = targeting.targeting_metrics(frame, "probability", 0.25)
        self.assertEqual(result["selected_n"], 2)
        self.assertEqual(result["selected_positive"], 2)
        self.assertAlmostEqual(result["precision_at_k"], 1.0)
        self.assertAlmostEqual(result["capture_rate"], 1.0)
        self.assertAlmostEqual(result["lift_at_k"], 4.0)

    def test_reported_twenty_percent_result_matches_predictions(self):
        frame = pd.read_csv(targeting.PREDICTIONS)
        result = targeting.targeting_metrics(
            frame, targeting.MODELS["Logistic_All"], 0.20
        )
        self.assertEqual(result["selected_n"], 924)
        self.assertEqual(result["selected_positive"], 641)
        self.assertAlmostEqual(result["precision_at_k"], 0.6937229437, places=9)
        self.assertAlmostEqual(result["capture_rate"], 0.3533627343, places=9)
        self.assertAlmostEqual(result["lift_at_k"], 1.7637542538, places=9)

    def test_saved_gain_curves_are_monotone_and_complete(self):
        path = Path(targeting.RESULTS) / "cumulative_gain_curves.csv"
        gains = pd.read_csv(path)
        for model, subset in gains.groupby("model"):
            ordered = subset.sort_values("contact_fraction")
            self.assertTrue(ordered["capture_rate"].is_monotonic_increasing, model)
            self.assertAlmostEqual(ordered.iloc[-1]["capture_rate"], 1.0, places=12)

    def test_threshold_grid_uses_lower_value_on_f1_ties(self):
        labels = pd.Series([1, 0])
        probabilities = pd.Series([0.80, 0.20])
        self.assertAlmostEqual(experiment.best_threshold(labels, probabilities), 0.205)

    def test_snapshot_respects_open_time_boundaries(self):
        cutoff = pd.Timestamp("2021-02-01")
        prediction_end = cutoff + pd.Timedelta(days=30)
        frame = pd.DataFrame(
            {
                "CustomerID": [1, 1, 1, 2, 2, 3],
                "InvoiceNo": ["H1", "P1", "E1", "H2", "E2", "P3"],
                "StockCode": ["A", "B", "C", "D", "E", "F"],
                "Quantity": [1, 1, 1, 1, 1, 1],
                "UnitPrice": [10.0, 999.0, 500.0, 20.0, 300.0, 40.0],
                "InvoiceDate": [
                    pd.Timestamp("2021-01-20"),
                    cutoff,
                    prediction_end,
                    pd.Timestamp("2021-01-25"),
                    prediction_end,
                    pd.Timestamp("2021-02-10"),
                ],
            }
        )
        frame["Amount"] = frame["Quantity"] * frame["UnitPrice"]
        snapshot = experiment.build_snapshot(frame, cutoff).set_index("CustomerID")

        self.assertEqual(set(snapshot.index), {1, 2})
        self.assertEqual(snapshot.loc[1, "label"], 1)
        self.assertEqual(snapshot.loc[2, "label"], 0)
        self.assertEqual(snapshot.loc[1, "monetary"], 10.0)
        self.assertEqual(snapshot.loc[2, "monetary"], 20.0)
        self.assertEqual(snapshot.loc[1, "frequency"], 1)
        self.assertEqual(snapshot.loc[1, "amount_std"], 0.0)
        self.assertEqual(snapshot.loc[1, "mean_purchase_interval"], 0.0)
        self.assertEqual(snapshot.loc[1, "prior30_orders"], 0.0)
        self.assertNotIn(999.0, snapshot["monetary"].tolist())
        self.assertNotIn(500.0, snapshot["monetary"].tolist())
        self.assertNotIn(300.0, snapshot["monetary"].tolist())


if __name__ == "__main__":
    unittest.main()
