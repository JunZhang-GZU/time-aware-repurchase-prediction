#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UCI Online Retail: leakage-safe repeat-purchase prediction."""
from pathlib import Path
import json, zipfile, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "online_retail.zip"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260827

def load_data():
    xlsx = ROOT / "data" / "raw" / "Online Retail.xlsx"
    if not xlsx.exists():
        with zipfile.ZipFile(RAW) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".xlsx")]
            if not names:
                raise FileNotFoundError("No xlsx file found in archive")
            with z.open(names[0]) as src, open(xlsx, "wb") as dst:
                dst.write(src.read())
    df = pd.read_excel(xlsx)
    df = df.dropna(subset=["CustomerID", "InvoiceDate"]).copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Amount"] = df["Quantity"] * df["UnitPrice"]
    return df.sort_values("InvoiceDate")

def build_snapshot(df, cutoff, obs_days=90, pred_days=30):
    cutoff = pd.Timestamp(cutoff)
    obs_start = cutoff - pd.Timedelta(days=obs_days)
    pred_end = cutoff + pd.Timedelta(days=pred_days)
    h = df[(df.InvoiceDate >= obs_start) & (df.InvoiceDate < cutoff)].copy()
    f = df[(df.InvoiceDate >= cutoff) & (df.InvoiceDate < pred_end)].copy()
    if h.empty:
        return pd.DataFrame()
    g = h.groupby("CustomerID")
    feat = g.agg(
        recency=("InvoiceDate", lambda x: (cutoff - x.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary=("Amount", "sum"),
        quantity=("Quantity", "sum"),
        unique_products=("StockCode", "nunique"),
        active_days=("InvoiceDate", lambda x: x.dt.date.nunique()),
        avg_unit_price=("UnitPrice", "mean"),
        amount_std=("Amount", "std"),
        tenure_days=("InvoiceDate", lambda x: (x.max() - x.min()).days),
    )
    basket = h.groupby(["CustomerID", "InvoiceNo"], as_index=False)["Amount"].sum()
    feat["avg_basket"] = basket.groupby("CustomerID")["Amount"].mean()
    recent = h[h.InvoiceDate >= cutoff - pd.Timedelta(days=30)].groupby("CustomerID")
    prior = h[(h.InvoiceDate >= cutoff - pd.Timedelta(days=60)) &
              (h.InvoiceDate < cutoff - pd.Timedelta(days=30))].groupby("CustomerID")
    feat["recent30_orders"] = recent["InvoiceNo"].nunique()
    feat["recent30_amount"] = recent["Amount"].sum()
    feat["prior30_orders"] = prior["InvoiceNo"].nunique()
    feat["prior30_amount"] = prior["Amount"].sum()
    feat["order_trend"] = feat["recent30_orders"] - feat["prior30_orders"]
    feat["amount_trend"] = feat["recent30_amount"] - feat["prior30_amount"]
    intervals = (h[["CustomerID","InvoiceNo","InvoiceDate"]].drop_duplicates()
        .sort_values(["CustomerID","InvoiceDate"])
        .groupby("CustomerID")["InvoiceDate"]
        .apply(lambda s: s.diff().dt.total_seconds().div(86400).mean()))
    feat["mean_purchase_interval"] = intervals
    labels = set(f["CustomerID"].unique())
    feat["label"] = feat.index.to_series().isin(labels).astype(int)
    feat["cutoff"] = cutoff
    return feat.reset_index().replace([np.inf, -np.inf], np.nan).fillna(0)

def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "threshold": float(threshold),
    }

def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 181)
    vals = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(vals))])

def main():
    df = load_data()
    cutoffs = pd.date_range("2011-04-01", "2011-11-01", freq="MS")
    panel = pd.concat([build_snapshot(df, c) for c in cutoffs], ignore_index=True)
    train = panel[panel.cutoff <= "2011-08-01"].copy()
    valid = panel[panel.cutoff == "2011-09-01"].copy()
    test = panel[panel.cutoff >= "2011-10-01"].copy()
    feature_all = [c for c in panel.columns if c not in ["CustomerID","label","cutoff"]]
    feature_rfm = ["recency","frequency","monetary"]
    specs = {
        "Logistic_RFM": (feature_rfm, Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))
        ])),
        "Logistic_All": (feature_all, Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))
        ])),
        "RandomForest_All": (feature_all, RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED)),
        "LightGBM_All": (feature_all, LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=15,
            max_depth=-1, colsample_bytree=0.8,
            class_weight="balanced", random_state=SEED, n_jobs=-1, verbosity=-1)),
    }
    rows = []
    fitted = {}
    for name, (cols, model) in specs.items():
        model.fit(train[cols], train.label)
        pv = model.predict_proba(valid[cols])[:,1]
        threshold = best_threshold(valid.label.values, pv)
        pt = model.predict_proba(test[cols])[:,1]
        row = {"model": name, **metrics(test.label.values, pt, threshold)}
        rows.append(row)
        fitted[name] = (cols, model)
    result = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    result.to_csv(OUT / "model_metrics.csv", index=False)

    cols, model = fitted["LightGBM_All"]
    imp = pd.DataFrame({
        "feature": cols,
        "importance_gain": model.booster_.feature_importance(importance_type="gain")
    }).sort_values("importance_gain", ascending=False)
    imp["importance_gain_pct"] = 100 * imp.importance_gain / imp.importance_gain.sum()
    imp.to_csv(OUT / "feature_importance.csv", index=False)

    summary = {
        "raw_rows": int(len(df)),
        "customers": int(df.CustomerID.nunique()),
        "date_min": str(df.InvoiceDate.min()),
        "date_max": str(df.InvoiceDate.max()),
        "panel_rows": int(len(panel)),
        "positive_rate": float(panel.label.mean()),
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "train_cutoffs": sorted(train.cutoff.astype(str).unique().tolist()),
        "validation_cutoffs": sorted(valid.cutoff.astype(str).unique().tolist()),
        "test_cutoffs": sorted(test.cutoff.astype(str).unique().tolist()),
    }
    (OUT / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(result.to_string(index=False))
    print("\nTop features:")
    print(imp.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
