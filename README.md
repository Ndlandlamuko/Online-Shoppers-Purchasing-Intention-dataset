# Online Shoppers Purchasing Intention — Classification

Predicting whether an e-commerce session ends in a purchase (`Revenue = 1`) from web-session behaviour. The same problem is tackled **two ways**:

- 🟦 **R** — a focused deep-dive on three classic learners (**Naive Bayes**, **Decision Trees**, **Logistic Regression**) with manual hyperparameter tuning, class balancing, and probability calibration (`h2o`, `rpart`, `caret`).
- 🐍 **Python** — a breadth-first **AutoML sweep** across 19 models using **PyCaret**, with an end-to-end preprocessing pipeline (outlier removal, transforms, encoding, multicollinearity filtering).

Both share the **same cleaning and feature-engineering logic**, so they make a clean side-by-side comparison of two common workflows: hand-rolled vs. AutoML.

The project walks the full applied-ML pipeline: cleaning, feature engineering, a stratified train/test split, model fitting with cross-validated tuning, threshold-based evaluation, and model comparison. The R track additionally covers class-balancing experiments (under/over/SMOTE) and probability re-calibration.

---

## Dataset

The [Online Shoppers Purchasing Intention dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (UCI) records one row per web session.

| | |
|---|---|
| Sessions (raw) | 12,330 |
| Sessions (after dropping 125 duplicates) | 12,205 |
| Target | `Revenue` (purchase: `1` / no purchase: `0`) |
| Class balance | 10,297 negative vs 1,908 positive (**≈ 15.6 % positive**) |
| Original predictors | 18 (10 numeric, 8 categorical) |

The target is moderately imbalanced, which is why evaluation leans on **F1, recall, and AUC** rather than raw accuracy, and why several resampling strategies are tested.

---

## Pipeline

1. **Load & inspect** — import from Excel, coerce numeric columns, check for missing values (none), drop exact duplicates.
2. **Encode** — map `Month` to an ordered integer, encode `VisitorType`, convert boolean flags, cast nominal columns (OS, browser, region, traffic type) to factors.
3. **Feature engineering** — ~30 derived features across 8 themed groups (see below).
4. **Split** — 70 / 30 **stratified** split (`caTools::sample.split`), seed `606`.
5. **Model** — Naive Bayes, two Decision Trees (an unpruned `h2o` grid-search tree and a pruned `rpart` tree), and Logistic Regression. 5-fold CV.
6. **Evaluate** — confusion matrices at a 0.5 threshold plus ROC/AUC on train and test.
7. **Balance & calibrate** — under-/over-/combined-/SMOTE resampling, baseline-LR technique selection, and Bayes prior-correction of probabilities.

### Engineered feature groups

| # | Group | Examples |
|---|-------|----------|
| 1 | Session breadth | `Total_Pages`, `Total_Duration`, `Avg_Duration_Per_Page` |
| 2 | Product focus | `Avg_ProductRelated_Duration`, `Product_Focus_Ratio`, `Non_Product_Ratio` |
| 3 | Engagement quality | `Engagement_Score = PageValues × (1 − BounceRate)`, `Exit_Minus_Bounce`, `Has_PageValue` |
| 4 | Temporal signals | `Season`, `Is_Q4`, `Near_Special_Day` |
| 5 | Visitor intent flags | `Is_New_Visitor`, `Is_Returning_Visitor`, `High_Intent` |
| 6 | Log transforms | `Log_*_Duration`, `Log_PageValues` (right-skew correction) |
| 7 | Revenue flags | *derived from the target — **dropped before modelling** to prevent leakage* |
| 8 | Device / traffic | `Is_Top_TrafficType`, `Is_Top_Browser`, `Is_Top_OS`, `Is_Top_Region` |

> ⚠️ Group 7 features are intentionally excluded from the predictor set. Because they are computed from `Revenue`, leaving them in would leak the answer and inflate every metric.

---

## R track — results

All models evaluated at a fixed **0.5 threshold** on the held-out test set (positive class = `1`).

### Test-set performance

| Model | Accuracy | Recall (Sens.) | Specificity | Precision | F1 | AUC |
|-------|:--------:|:--------------:|:-----------:|:---------:|:--:|:---:|
| Naive Bayes | 0.844 | **0.832** | 0.847 | 0.501 | 0.626 | 0.890 |
| Decision Tree — `h2o`, unpruned | 0.859 | 0.556 | 0.915 | 0.547 | 0.552 | 0.739 |
| Decision Tree — `rpart`, pruned | 0.897 | 0.696 | 0.934 | 0.660 | **0.677** | 0.861 |
| Logistic Regression | **0.898** | 0.596 | **0.954** | **0.705** | 0.646 | **0.918** |

### Train vs test AUC (overfitting check)

| Model | Train AUC | Test AUC | Gap |
|-------|:---------:|:--------:|:---:|
| Naive Bayes | 0.882 | 0.890 | ~0 |
| Decision Tree — `h2o`, unpruned | **0.9995** | 0.739 | **0.26** |
| Decision Tree — `rpart`, pruned | 0.862 | 0.861 | ~0 |
| Logistic Regression | 0.926 | 0.918 | ~0 |

### Takeaways

- **Logistic Regression** gives the best test AUC (0.918) and the best precision, making it the strongest overall ranker and the most reliable single model here.
- **Naive Bayes** has the highest recall (0.832) — useful if the cost of missing a buyer is high — at the expense of precision.
- The **unpruned `h2o` tree** is a textbook overfitting case: a near-perfect 0.9995 train AUC collapses to 0.739 on test. Pruning the `rpart` tree (`maxdepth = 5`) closes that gap almost entirely (0.862 → 0.861), trading a little training fit for genuine generalization.

---

## Python (PyCaret) — AutoML model sweep

The Python track (`python/main.ipynb`) reuses the same engineered features, then hands a fully configured `setup()` to PyCaret and compares **19 models** on a stratified holdout. Preprocessing inside the pipeline includes iForest outlier removal, robust scaling, Yeo-Johnson transforms, one-hot encoding, and multicollinearity filtering.

![PyCaret leaderboard — holdout AUC](images/pycaret_model_leaderboard.png)

Holdout leaderboard (top of the table, by AUC):

| Model | Accuracy | AUC | Recall | Precision | F1 |
|-------|:--------:|:---:|:------:|:---------:|:--:|
| Gradient Boosting | 0.898 | **0.927** | 0.607 | 0.698 | 0.649 |
| CatBoost | 0.899 | 0.926 | 0.591 | 0.712 | 0.646 |
| LightGBM | 0.893 | 0.923 | 0.584 | 0.683 | 0.630 |
| Random Forest | 0.893 | 0.922 | 0.570 | 0.689 | 0.624 |
| Logistic Regression | 0.890 | 0.912 | 0.598 | 0.665 | 0.630 |
| Naive Bayes | 0.829 | 0.870 | **0.813** | 0.473 | 0.598 |
| Decision Tree | 0.857 | 0.730 | 0.546 | 0.543 | 0.544 |
| Dummy (baseline) | 0.844 | 0.500 | 0.000 | 0.000 | 0.000 |

### Cross-method consistency

The two implementations tell the same story from different angles:

- **Gradient boosting / CatBoost win** the breadth sweep (AUC ≈ 0.927), edging out logistic regression — the gain a richer model buys on this dataset is real but modest (~0.015 AUC).
- **Logistic Regression** is the best of the "classic" trio in *both* tracks (R test AUC 0.918, PyCaret holdout AUC 0.912) — strong evidence it's a genuinely solid baseline, not a fluke of one split.
- **Naive Bayes** again posts the **highest recall** (0.81) at low precision in both tracks — same recall-heavy behaviour, independently reproduced.
- **A single Decision Tree is the weakest** real model in both (AUC ≈ 0.73), which is exactly why the R track pairs it with pruning and the Python track prefers tree *ensembles*.

> The PyCaret `compare_models` leaderboard above uses a holdout evaluation; the `dt` model was additionally validated with 10-fold stratified CV (mean AUC 0.730 ± 0.023), matching the holdout figure closely.

---

## Visuals

**Combined ROC — test set (NB vs DT vs LR)**

![ROC comparison on the test set](images/roc_comparison_test.jpeg)

Logistic regression (blue) sits highest across most of the curve; the pruned decision tree (red) is more piecewise but competitive; Naive Bayes (green) trails slightly.

**Decision-tree overfitting: train vs test**

| Unpruned `h2o` tree — train (AUC 0.9995) | Unpruned `h2o` tree — test (AUC 0.739) |
|---|---|
| ![h2o DT train ROC](images/dt_h2o_roc_train.png) | ![h2o DT test ROC](images/dt_h2o_roc_test.png) |

The near-right-angle on the left is the giveaway — the tree has memorised the training data. The coarse, collapsed curve on the right is what that memorisation costs on unseen sessions.

**Cost-complexity pruning (`rpart`)**

![rpart cp pruning plot](images/rpart_cp_pruning.png)

Cross-validated relative error keeps falling as the tree grows to ~5 leaves, where it settles near the 1-SE line — the basis for choosing the pruned tree.

**Naive Bayes ROC** — [train](images/nb_roc_train.png) · [test](images/nb_roc_test.png)
**Pruned `rpart` ROC** — [train](images/rpart_roc_train.png) · [test](images/rpart_roc_test.png)

---

## Repository structure

```
online-shoppers-purchase-prediction/
├── README.md
├── R/
│   └── shoppers_classification.R     # full R pipeline (NB, DT, LR + balancing)
├── python/
│   ├── main.ipynb                    # PyCaret AutoML sweep (19 models)
│   └── requirements.txt
├── datasets/
│   └── Online Shoppers Purchasing Intention dataset.xlsx   # (add this yourself)
└── images/
    ├── roc_comparison_test.jpeg
    ├── pycaret_model_leaderboard.png
    ├── nb_roc_train.png   ├── nb_roc_test.png
    ├── dt_h2o_roc_train.png   ├── dt_h2o_roc_test.png
    ├── rpart_roc_train.png    ├── rpart_roc_test.png
    └── rpart_cp_pruning.png
```

---

## Running it

Place the dataset Excel file in `datasets/` first (see `datasets/README.md`).

### R track

**Requirements:** R ≥ 4.5 and Java 8+ (for `h2o`).

```r
install.packages(c(
  "dplyr", "caTools", "caret", "pROC",
  "h2o", "readxl", "rpart", "rpart.plot"
))
```

Open `R/shoppers_classification.R`, adjust the parameters at the top if you like, then run top to bottom:

```r
seed       <- 606      # reproducibility
train_frac <- 0.7      # train proportion
metric     <- "F1"     # grid-search selection metric
folds      <- 5        # CV folds
```

`h2o.init()` starts a local cluster; `h2o.shutdown(prompt = FALSE)` releases it at the end.

### Python track

**Requirements:** Python 3.10.

```bash
cd python
pip install -r requirements.txt
jupyter lab            # then open main.ipynb and run all cells
```

PyCaret's `setup()` handles the split, preprocessing, and stratified 10-fold CV; `compare_models()` produces the leaderboard and `evaluate_model()` opens interactive diagnostics.

---

## Notes & caveats

- A handful of `TrafficType` levels appear only in the test split (e.g. level `17`), producing a harmless "levels not trained on" warning from `h2o`.
- The 0.5 threshold is a baseline, not an optimum. On an imbalanced target, tuning the threshold (or correcting probabilities after resampling) materially changes the precision/recall trade-off.
- After over-/under-sampling, predicted probabilities are miscalibrated; the script includes a Bayes prior-correction so probabilities reflect the real-world class prior again.

## License

Released under the MIT License — see `LICENSE`. The dataset is distributed by the UCI Machine Learning Repository under its own terms.
