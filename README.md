# time-aware-repurchase-prediction
Code for “E-Commerce Repeat-Purchase Prediction under Time-Boundary Constraints”, using the UCI Online Retail dataset with rolling observation and prediction windows.

《时间边界约束下的电商用户复购预测研究》实验代码与归档结果。

## Contents / 文件说明

- `run_experiment.py`: data cleaning, 90-day observation / 30-day prediction windows, temporal splits, four classifiers, validation-selected thresholds and feature importance.
- `run_robustness.py`: model fitting, test predictions, monthly metrics and 2,000 paired customer-cluster bootstrap replicates.
- `run_calibration.py`: Brier score, log loss, 10-bin ECE, Brier difference bootstrap and Figure 3.
- `run_marketing_targeting.py`: monthly 10%, 20% and 30% contact budgets, customer-cluster bootstrap and Figure 4.
- `test_marketing_targeting.py`: five existing tests covering time boundaries, threshold ties, budget metrics and saved results.
- `results/`: archived metrics and predictions derived from the public UCI dataset. CustomerID is the dataset identifier.
- `figures/`: archived PNG and PDF figures. Scripts for Figures 3 and 4 are included; the original plotting scripts for Figures 1 and 2 were not present in the supplied experiment directory.

## Environment / 环境

The source environment was inspected on 2026-09-12: Python 3.8.18 on Linux. Direct numerical and plotting dependencies are pinned in `requirements.txt` to the installed versions. Use an isolated environment with a matching Python version for closest reproduction:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. LightGBM may also require the platform's OpenMP runtime. Models use all available CPU threads; run training on an appropriate compute node when using a cluster.

## Data / 数据

Download **Online Retail**, DOI [10.24432/C5BW33](https://doi.org/10.24432/C5BW33), from the [UCI dataset page](https://archive.ics.uci.edu/dataset/352/online+retail).

Create `data/raw/` and save the downloaded archive as `data/raw/online_retail.zip`. Alternatively, place the extracted Excel file at `data/raw/Online Retail.xlsx`. The original transaction files are not bundled in this repository.

SHA-256 of the archive used in the source experiment:

```text
f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b
```

The archived summary contains 397,884 **cleaned** transaction rows, 4,338 customers and 16,155 customer-cutoff samples. The legacy `raw_rows` key in `dataset_summary.json` refers to cleaned rows, not the original download's row count.

## Run / 运行顺序

Run these commands from the repository root. They overwrite the corresponding archived results and generated figures:

```bash
python run_experiment.py
python run_robustness.py
python run_calibration.py
python run_marketing_targeting.py
python -m unittest -v test_marketing_targeting.py
```

`run_robustness.py` generates `results/test_predictions.csv`, which is required by the calibration and marketing scripts. Bootstrap analyses can take substantially longer than the unit tests.

## Archived results / 归档结果

Training cutoffs: April–August 2011 (9,620 samples); validation: September (1,923); test: October–November (4,612). Random seed: 20260827. Thresholds are chosen on validation data, not test data. PR-AUC in these scripts is computed using scikit-learn's `average_precision_score`.

| Model | ROC-AUC | Average precision | F1 |
|---|---:|---:|---:|
| Logistic_All | 0.6894 | 0.6313 | 0.5859 |
| RandomForest_All | 0.6760 | 0.6274 | 0.5695 |
| LightGBM_All | 0.6758 | 0.6209 | 0.5702 |
| Logistic_RFM | 0.6754 | 0.6194 | 0.5509 |

These are archived experiment results, not newly generated training results from the repository preparation step. All five existing tests passed in the source environment on 2026-09-12. A complete fresh training run was not performed during upload preparation; library versions and hardware may affect numerical reproducibility.

Conclusions are limited to the current dataset, features and time windows. The supplementary Cox, BG/NBD and window-sensitivity experiments are implemented and archived separately in [revision_20260913](revision_20260913/README.md). The code preserves the supplied experiment implementation, including its feature construction and missing-value handling.

## Publication scope

This repository contains experiment materials only. Submission documents, author contact details, server connection settings, raw data archives and temporary files are excluded. No software license is added by this upload; dataset terms are available from UCI.

## Supplementary experiments / 新增实验

See [reproduction instructions](revision_20260913/README.md), [experiment code](revision_20260913/run_revision_experiments.py), [baseline metrics](revision_20260913/results/additional_baselines.csv) and [window sensitivity](revision_20260913/results/window_sensitivity.csv). These experiments were run on 2026-09-13; this upload publishes the archived outputs without rerunning or changing the original experiments.
