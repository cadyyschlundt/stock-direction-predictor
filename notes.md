# Project 2: ML Direction Predictor - Notes

## Current Status
- Repo: stock-direction-predictor (already set up with full folder structure)
- Data: data/raw/spy_features.csv (loaded from CSV, no parquet yet)
- Completed: Notebook 01 EDA
- Next step: Notebook 02 baseline model
- Pending decision: how to handle correlated price-level features before modeling

## Tech Stack Decisions
- XGBoost and LightGBM excluded: failed to build on Windows ARM64 (CMake/C++ 
  build tool issue), not worth debugging
- Using scikit-learn native models only:
  - LogisticRegression (baseline)
  - RandomForestClassifier (mid complexity)
  - GradientBoostingClassifier (primary gradient boost)
  - HistGradientBoostingClassifier (faster comparison)



## Notebook 01: EDA

### Dataset
- Source: SPY only (foundation phase, single ticker)
- Date range: 2014-10-16 to 2026-06-26
- Rows: 2939 (after dropping last row for label creation)
- Columns: 24 original (Date, Close, High, Low, Open, Volume, RSI, MACD,
  MACD_Signal, MACD_Hist, BB_Upper, BB_Middle, BB_Lower, SMA_20, SMA_50,
  SMA_200, EMA_12, EMA_26, ATR, OBV, Stoch_K, Stoch_D, Williams_R, ROC)
- No missing values in original CSV
- Date was loaded as `str`, converted to `datetime64` and set as index

### Label
- Binary target: 1 if Close_t+1 > Close_t, else 0
- Created via `Close.shift(-1) > Close`, last row dropped (no future value)
- Class balance: 1615 up days (55%) vs 1324 down days (45%)
- Baseline to beat: ~55% accuracy (always predicting "up")

### Distribution findings
- RSI, Stoch_K, Stoch_D, Williams_R all correctly bounded in expected ranges
- MACD, MACD_Signal, MACD_Hist, ROC center around zero as expected
- Price-derived indicators (BB_Upper/Middle/Lower, SMA_20/50/200, EMA_12/26)
  all show similar multi-modal shapes, since they're all derived from price
- OBV is on a much larger scale (billions) than other features, flag for
  scaling/normalization later

### Correlation findings
- 12 of 18 features are highly correlated (>0.9): Close, High, Low, Open,
  BB_Upper, BB_Middle, BB_Lower, SMA_20, SMA_50, SMA_200, EMA_12, EMA_26
  -> all essentially measuring price level, redundant for modeling
- Moderate cluster: Stoch_K, Stoch_D, Williams_R, ROC, RSI (~0.4-0.6 correlation)
  -> momentum-style indicators, related but not redundant
- Volume is mostly independent/slightly negative correlation with other features
- ATR, OBV, MACD_Hist show independent behavior, likely useful standalone signals
- ACTION ITEM for Project 2 modeling: need to address price-level redundancy,
  either drop most price-derived features or use tree-based feature importance
  / PCA to let the model decide

### Time series sanity check
- Close price chart matches real SPY history (2020 COVID crash, 2022 bear
  market, 2026 highs all visible)
- RSI oscillates correctly between ~18-87, no stuck/flat values
- SMA_50 correctly lags and smooths price as expected
- Volume spikes align with major market events (largest spike ~2020 COVID crash)

### Return distribution
- Added `Daily_Return` column via `Close.pct_change()` (permanent column)
- Mean daily return: 0.0006 (~0.06%/day, consistent with long-term upward bias)
- Std: 0.011 (~1.1% typical daily move)
- Min: -0.109 (confirmed via `.idxmin()` -> 2020-03-16, matches real-world
  COVID crash day, S&P's worst day since 1987 Black Monday)
- Max: 0.105 (likely a COVID-era recovery day)
- Distribution shows expected fat-tailed shape (tight peak near 0, long tails)

### Decisions made
- Including raw price columns in correlation check (not dropping permanently,
  just temp exclusion for histogram grid)
- Adding label and Daily_Return as permanent columns in df
- Saving redundancy issue for Project 2 feature selection step, not fixing now

### Open questions / stretch ideas (not in current roadmap scope)
- Considered adding news sentiment data, decided to revisit near end of
  5-week timeline if time allows, not part of core 5 projects

## Feature Selection Decisions
- Dropped: High, Low, Open, BB_Middle, BB_Lower, SMA_20, SMA_200, EMA_12, EMA_26
- Reasoning: highly correlated price-level features, redundant for modeling
- Kept one representative per group:
  - SMA_50 (medium-term trend)
  - BB_Upper (price position relative to volatility band)
  - EMAs dropped since MACD already captures EMA_12 - EMA_26
  - High/Low/Open dropped since Close and ATR cover them
- Final feature set: 15 features



## Notebook 02: Baseline Model (Logistic Regression)

### Results
| Fold | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| 1    | 0.429    | 0.417     | 0.018  | 0.035 |
| 2    | 0.509    | 0.555     | 0.703  | 0.620 |
| 3    | 0.538    | 0.538     | 1.000  | 0.699 |
| 4    | 0.540    | 0.539     | 0.981  | 0.696 |
| 5    | 0.550    | 0.560     | 0.949  | 0.704 |
| Avg  | 0.513    | 0.522     | 0.730  | 0.551 |

### Interpretation
- Baseline to beat: 55% accuracy (always predicting "up")
- Logistic Regression average accuracy 51.3% -- below baseline as expected
- Fold 1 nearly always predicted "down" -- too little training data (493 rows)
- Folds 2-5 trend toward always predicting "up", picking up on the 55% upward bias
- Logistic Regression hurt by remaining feature correlations
- Tree-based models expected to outperform this baseline

### Decisions made
- max_iter=1000 to ensure convergence
- random_state=42 for reproducibility
- Scaler fitted on training data only per fold (no leakage)

### Random Forest Results
| Fold | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| 1    | 0.474    | 0.643     | 0.162  | 0.259 |
| 2    | 0.474    | 0.551     | 0.423  | 0.479 |
| 3    | 0.474    | 0.625     | 0.057  | 0.105 |
| 4    | 0.497    | 0.530     | 0.546  | 0.538 |
| 5    | 0.542    | 0.570     | 0.772  | 0.655 |
| Avg  | 0.492    | 0.584     | 0.392  | 0.407 |

### Interpretation
- Average accuracy 49.2% -- below baseline (55%) and below Logistic Regression
- Precision decent at 0.584 -- when it predicts "up" it's right 58% of the time
- Low recall -- model too conservative, missing too many up days
- Fold 3 nearly never predicted "up" (recall 0.057), suspicious
- Likely causes: default hyperparameters untuned, stock data is noisy
- Gradient boosting models expected to outperform


### Model Comparison Summary
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic Regression | 0.513 | 0.522 | 0.730 | 0.551 |
| Random Forest | 0.492 | 0.584 | 0.392 | 0.407 |
| Gradient Boosting | 0.484 | 0.458 | 0.343 | 0.357 |
| Hist Gradient Boosting | 0.500 | 0.564 | 0.500 | 0.497 |
| Baseline (always up) | 0.550 | - | - | - |

### Key Takeaways
- No model beat the 55% baseline on accuracy
- Logistic Regression highest accuracy but only because it predicts "up" too often
- Hist Gradient Boosting most balanced -- best precision/recall tradeoff
- Gradient Boosting Fold 3 predicted "down" every single day (likely COVID period)
- Honest results from proper walk-forward validation -- no data leakage
- Selected model for downstream use: Hist Gradient Boosting (most balanced)

### Confusion Matrix (Fold 5 - Hist Gradient Boosting)
|                  | Predicted Down | Predicted Up |
|------------------|---------------|--------------|
| Actual Down      | 33            | 180          |
| Actual Up        | 40            | 236          |

### Interpretation
- Model heavily biased toward predicting "up" (market upward bias)
- False Positives high (180) -- many bad trades
- When predicting "up", correct 236/416 times (~57%) -- slightly above baseline
- Confusion matrix saved to results/confusion_matrix.png

## Model Finalization

- Retrained HistGradientBoostingClassifier on full dataset (all data, not
  just folds), same hyperparameters as validation (max_iter=100,
  learning_rate=0.1, random_state=42)
- Reasoning: walk-forward folds are for validation only, final deployable
  model should use all available data
- Saved as models/hgb_v1.joblib, scaler as models/scaler_v1.joblib
  (versioned naming to leave room for v2 later)
- No leakage concern: retraining on everything happens after evaluation,
  not as part of it, and this model's accuracy is never re-reported as
  a validation metric

## Equity Curve (Out-of-Sample, Stitched Across All 5 Folds)

- Collected predictions, actuals, and dates from all 5 walk-forward folds
  into oos_results_df (2445 rows, 2016-10-03 to 2026-06-25)
- Strategy: long SPY when HGB predicts "up," cash (0% return) otherwise,
  no shorting
- BUG FOUND: Daily_Return (from Notebook 01) is backward-looking
  (today vs yesterday), but the label is forward-looking (tomorrow vs
  today) -- caused ~25% mismatch rate between actual and daily_return
- FIX: added Next_Day_Return column to df via Daily_Return.shift(-1),
  re-merged into oos_results_df -- mismatch dropped to 0
- Dropped last row (2026-06-25) after fix, no next-day data to shift in
  -- 2444 rows for final calculations

### Results
| Metric | Strategy (Long/Cash) | Buy & Hold |
|--------|----------------------|------------|
| Total Return | ~104% | ~297% |
| Sharpe Ratio (annualized, 0% risk-free) | 0.564 | 0.877 |
| Max Drawdown | -28.95% | -33.72% |

- Strategy underperforms buy-and-hold on total return AND risk-adjusted
  return (Sharpe), despite a smaller max drawdown
- Consistent with model's known "up" bias -- sitting out too often on
  days that turn out positive costs more than the downside protection
  is worth
- Honest result, no leakage -- model does not yet add value over
  simple buy-and-hold baseline
- Chart saved to results/equity_curve.png

## metrics.json

- Saved to results/metrics.json: model_comparison (all 4 models),
  baseline accuracy, selected_model, equity_curve_metrics
  (strategy + buy-and-hold Sharpe/drawdown/total return), metadata
  (training_date, model_type, hyperparameters, features list)
- Structured for reuse in Project 3's /model/info endpoint
## Notebook 02b: Logistic Regression Tuning

### Approach
- Created new TimeSeriesSplit folds (5 splits) separate from notebook 02's folds, for practice
- Manually tested 5 C values: 0.01, 0.1, 1.0, 10, 100
- Scaler fit on training data only per fold (no leakage), same as notebook 02
- Tracked both blended metrics (accuracy, precision, recall, f1) and per-class
  metrics (precision/recall separately for up and down classes) to check
  whether tuning actually reduced the model's bias toward predicting "up"

### C value tuning (class_weight=None)
| C     | Accuracy | Recall (Down) | Recall (Up) |
|-------|----------|----------------|-------------|
| 0.01  | 0.530    | 0.213          | 0.792       |
| 0.10  | 0.519    | 0.244          | 0.750       |
| 1.00  | 0.513    | 0.257          | 0.730       |
| 10.00 | 0.511    | 0.259          | 0.725       |
| 100.00| 0.510    | 0.257          | 0.725       |

- Lower C (more regularization) gave the highest accuracy (0.530 at C=0.01)
  but the worst down-day recall (0.213), meaning it leaned hardest into
  predicting "up"
- Higher C slightly improved down-day recall but not meaningfully, and
  accuracy dropped
- Adjusting C alone did not fix the up-bias problem

### class_weight='balanced' tuning
| C     | class_weight | Accuracy | Recall (Down) | Recall (Up) |
|-------|--------------|----------|----------------|-------------|
| 0.01  | balanced     | 0.494    | 0.374          | 0.596       |
| 1.00  | balanced     | 0.483    | 0.374          | 0.582       |
| 100.00| balanced     | 0.484    | 0.371          | 0.586       |

- class_weight='balanced' had a much bigger effect than C on the up/down
  bias, raising down-day recall from ~0.21-0.26 up to ~0.37 consistently
  across all C values
- Cost: accuracy dropped by about 3-4 points every time balancing was used
- C value barely mattered once class_weight was set to balanced; class_weight
  was doing almost all the work

### Key takeaway
- Confirmed a real tradeoff for Logistic Regression on this feature set:
  cannot get both higher accuracy AND better down-day recall at the same time
  using C and class_weight alone
- This is an honest finding about the model's limits, not a bug or failure
- Decision: documenting this tradeoff and moving on to tuning Random Forest
  and Gradient Boosting next, to see if tree-based models handle the
  up/down imbalance differently

### Bugs encountered and fixed
- Typo caused `fold_num` to print as a stray `_num` variable (NameError),
  fixed by correcting the variable name in the print statement
- pandas groupby silently drops rows where the group column is None/NaN by
  default; had to add `dropna=False` to see both balanced and unbalanced
  results side by side in the same table

## Notebook 02b: Random Forest Tuning (GridSearchCV)

### Approach
- Used GridSearchCV instead of manual loops, since Random Forest has more
  hyperparameters to test and GridSearchCV automates the process
- Param grid: n_estimators [100, 200], max_depth [5, 10, None],
  class_weight ['balanced', None] -- 12 total combinations
- Used existing TimeSeriesSplit (tscv) as the cross validation strategy,
  same folds as Logistic Regression tuning
- No feature scaling needed for Random Forest, since it splits on thresholds
  rather than distances
- Built a custom scorer using make_scorer(recall_score, pos_label=0) to
  track recall on the down class specifically
- Tracked both recall_down and accuracy simultaneously (multi-metric scoring),
  with refit='recall_down' so GridSearchCV picks its official winner based
  on down-day recall, not accuracy

### Results (all 12 combinations, sorted by recall_down)
| n_estimators | max_depth | class_weight | recall_down | accuracy |
|--------------|-----------|--------------|-------------|----------|
| 200          | None      | None         | 0.617       | 0.488    |
| 100          | None      | None         | 0.615       | 0.491    |
| 200          | 5         | balanced     | 0.585       | 0.479    |
| 100          | 5         | balanced     | 0.580       | 0.484    |
| 200          | None      | balanced     | 0.578       | 0.484    |
| 100          | None      | balanced     | 0.568       | 0.486    |
| 100          | 10        | None         | 0.560       | 0.500    |
| 200          | 10        | None         | 0.560       | 0.491    |
| 200          | 10        | balanced     | 0.560       | 0.490    |
| 100          | 10        | balanced     | 0.553       | 0.490    |
| 100          | 5         | None         | 0.515       | 0.501    |
| 200          | 5         | None         | 0.499       | 0.501    |

### Key takeaways
- Official winner (best recall_down): n_estimators=200, max_depth=None,
  class_weight=None -- recall_down 0.617, accuracy 0.488
- Unlimited tree depth (max_depth=None) drove most of the recall_down
  improvement, more so than class_weight='balanced'
- This differs from Logistic Regression, where class_weight was the main
  driver of recall_down improvement
- Same fundamental tradeoff as Logistic Regression: higher recall_down
  consistently came with lower accuracy
- Every single Random Forest combination fell below the 55% baseline
  accuracy, and below Logistic Regression's best accuracy results
- Decision: documenting Random Forest as a separate finding from Logistic
  Regression, not declaring one model an overall winner yet. Moving on to
  Gradient Boosting tuning next.

## Notebook 02b: Gradient Boosting Tuning (GridSearchCV)

### Approach
- Used GridSearchCV, same overall structure as Random Forest tuning
- GradientBoostingClassifier has no class_weight parameter, so this grid
  skipped class weighting entirely (would require a more complex
  sample_weight workaround, not done here)
- Param grid: n_estimators [100, 200], max_depth [3, 5],
  learning_rate [0.01, 0.1] -- 8 total combinations
- Reused existing tscv and recall_down_scorer from Random Forest section
- Same multi-metric setup: tracked recall_down and accuracy together,
  refit='recall_down' so GridSearchCV picks its winner based on down-day
  recall

### Results (all 8 combinations, sorted by recall_down)
| n_estimators | max_depth | learning_rate | recall_down | accuracy |
|--------------|-----------|----------------|-------------|----------|
| 200          | 3         | 0.10           | 0.682       | 0.480    |
| 200          | 5         | 0.10           | 0.648       | 0.484    |
| 100          | 3         | 0.10           | 0.645       | 0.494    |
| 100          | 5         | 0.10           | 0.622       | 0.481    |
| 200          | 5         | 0.01           | 0.574       | 0.489    |
| 200          | 3         | 0.01           | 0.525       | 0.488    |
| 100          | 5         | 0.01           | 0.520       | 0.499    |
| 100          | 3         | 0.01           | 0.495       | 0.500    |

### Key takeaways
- Official winner (best recall_down): n_estimators=200, max_depth=3,
  learning_rate=0.10 -- recall_down 0.682, accuracy 0.480
- Highest recall_down of all three tuned models so far (Logistic Regression
  0.374, Random Forest 0.617, Gradient Boosting 0.682)
- learning_rate=0.10 clearly outperformed 0.01 for recall_down across every
  matching pair of settings
- Shallower trees (max_depth=3) outperformed deeper ones (max_depth=5) at
  the same learning_rate, opposite of the Random Forest pattern where
  unlimited depth helped
- Same tradeoff as the other two models: accuracy stayed low across all
  8 combinations, never breaking 0.500, well below the 55% baseline

### Caveat: Fold 3 anomaly
- Raw per-fold output showed recall_down at or near 1.000 in Fold 3 across
  almost every single combination, regardless of hyperparameters
- Likely means Fold 3's validation window contains an unusually high
  proportion of down days (possibly overlapping a bear market period like
  2020 COVID crash or 2022), rather than every combination being
  genuinely good at catching down days
- This may be inflating the recall_down averages reported above; worth
  revisiting in notebook 03 by looking at fold-by-fold detail rather than
  just averages, to see how much this one fold is driving the results

### Decision
- Documenting Gradient Boosting as a separate finding, same as Logistic
  Regression and Random Forest, not declaring an overall winner across all
  three models yet
### Decision: Final Model Selection Criterion

- Tested four model families (Logistic Regression, Random Forest, Gradient
  Boosting, HistGradientBoosting) across 40+ hyperparameter combinations via
  walk-forward validation
- Best accuracy achieved: Logistic Regression at 0.530 (C=0.01, no
  class_weight)
- No model beat the 55% baseline (always predict "up")
- Considered continuing to search hyperparameters, decided against it due to
  multiple comparisons risk -- with 40+ combinations already tested, finding
  one that clears baseline would likely reflect overfitting to these specific
  validation folds rather than a genuine improvement
- Consistent with known difficulty of predicting daily direction on a liquid,
  closely watched market like SPY
- Decision: document as an honest finding rather than keep searching. Moving
  forward, equity curve / Sharpe ratio will be the deciding factor for model
  comparison in Notebook 03, since that measures whether predictions
  translate into real trading value, not just classification accuracy
- Alternatives considered but deferred: new feature engineering (lagged
  returns, day-of-week, volatility regime), ensembling the four tuned models

### Note: Macro vs Micro Averaging (Notebook 03 equity curves)

- Built generate_oos_predictions() function to stitch out-of-sample
  predictions across all 5 walk-forward folds for a given model + config,
  reusable across all 8 configs (accuracy-best and recall_down-best per
  model family)
- Sanity-tested against HGB's recall_down-best config from Notebook 02b's
  GridSearchCV results (expected: recall_down 0.569, accuracy 0.493)
- Function returned recall_down 0.595, accuracy 0.479 -- close but not
  identical
- Root cause: GridSearchCV reports a macro-average (recall_down computed
  per fold, then the 5 fold scores averaged equally). The function instead
  pools all 5 folds' predictions into one combined set first, then computes
  recall_down once across everything (micro-average). These can legitimately
  differ when folds vary in size or class balance, confirmed here since
  fold 5's recall_down (0.047) was far lower than the other folds (0.925,
  0.538, 0.867, 0.467), so pooling weights it differently than a plain
  average would
- Not a bug, both numbers are valid, they answer different questions:
  macro = "average performance across a typical fold," micro = "performance
  across the full out-of-sample period as one continuous stretch"
- Decision: use micro-averaged (pooled/stitched) numbers going forward for
  the equity curve work, since the equity curve itself simulates one
  continuous trading period, not five separate averaged periods. Expect
  the 8 equity curve configs' recall_down/accuracy to differ slightly from
  the original tuning tables for this reason

## Notebook 03: Equity Curve Comparison (All 8 Configs)

### Approach
- Built generate_oos_predictions() to stitch walk-forward predictions for
  any model/config combination, reused across all 8 configs (accuracy-best
  and recall_down-best per model family, from Notebook 02b tuning)
- Ran each config's predictions through the same long/cash equity curve
  logic from Notebook 02 (long when predicting "up," cash otherwise)

### Results (Sharpe / Total Return / Max Drawdown)
| Config        | Sharpe | Total Return | Max Drawdown |
|---------------|--------|--------------|--------------|
| lr_accuracy   | 0.725  | 1.924        | -0.337       |
| lr_recall     | 0.439  | 0.732        | -0.337       |
| rf_accuracy   | 0.624  | 1.182        | -0.304       |
| rf_recall     | 0.488  | 0.747        | -0.301       |
| gb_accuracy   | 0.482  | 0.800        | -0.304       |
| gb_recall     | 0.553  | 0.872        | -0.283       |
| hgb_accuracy  | 0.788  | 2.154        | -0.283       |
| hgb_recall    | 0.325  | 0.424        | -0.283       |
| Buy & Hold    | 0.875  | 2.965        | -0.337       |

### Key findings
- No config beats buy-and-hold on Sharpe or total return, confirming the
  Notebook 02b finding with real trading simulation instead of just
  classification metrics
- hgb_accuracy is the standout: closest Sharpe/return to buy-and-hold, and
  the only config with both a better Sharpe-adjusted profile AND a smaller
  drawdown (-0.283 vs -0.337)
- Pattern across 3 of 4 model families (LR, RF, HGB): the accuracy-leaning
  config beat the recall_down-leaning config on Sharpe. Leaning toward
  accuracy means leaning toward predicting "up" more often given the class
  imbalance, which captures more of the market's upward drift
- Gradient Boosting is the exception: gb_recall (0.553) beat gb_accuracy
  (0.482) on Sharpe, the only model family where being more cautious about
  "up" calls helped rather than hurt. Worth investigating further, possibly
  in Notebook 04
- Every config has a smaller max drawdown than buy-and-hold, consistent
  with downside protection, but this doesn't offset the Sharpe/return gap
  for 7 of the 8 configs

### Decision: winning config per model family (for ROC curves, confusion
matrices, feature importance)
- Logistic Regression: lr_accuracy (C=0.01, class_weight=None)
- Random Forest: rf_accuracy (n_estimators=100, max_depth=5, class_weight=None)
- Gradient Boosting: gb_recall (n_estimators=200, max_depth=3, learning_rate=0.10)
- HistGradientBoosting: hgb_accuracy (class_weight=None, learning_rate=0.01,
  max_depth=3, max_iter=100)
- Selection based on equity curve Sharpe/total return, not raw classification
  accuracy or recall_down, per earlier session decision to let backtest
  performance be the deciding factor
## Notebook 03: Confusion Matrices (4 Winning Configs)

### Approach
- Built confusion matrices for the 4 winning configs (lr_accuracy,
  rf_accuracy, gb_recall, hgb_accuracy), one figure each, showing raw
  count and row-normalized percentage per cell
- Row normalization confirms recall per class directly: down-recall shown
  in the "Actual Down" row, up-recall shown in the "Actual Up" row
- Cross-checked against earlier printed accuracy/recall_down values, all
  four matrices match exactly, confirming no issues in the merge or
  plotting steps

### Results (Actual Down row, Predicted Down / Predicted Up)
| Config       | Predicted Down | Predicted Up | Recall (Down) |
|--------------|-----------------|----------------|-----------------|
| lr_accuracy  | 225 (20.7%)     | 863 (79.3%)    | 0.207           |
| rf_accuracy  | 610 (56.1%)     | 478 (43.9%)    | 0.561           |
| gb_recall    | 716 (65.8%)     | 372 (34.2%)    | 0.658           |
| hgb_accuracy | 333 (30.6%)     | 755 (69.4%)    | 0.306           |

### Key finding: gb_recall is not "balanced," it's inverted
- Predicted expectation was that gb_recall would look more balanced between
  predicted-up and predicted-down compared to the other three configs
- Actual result: gb_recall predicts "down" for 1584 of 2445 rows (65%),
  even though only ~44.5% of days were actually down. It swung from an
  up-bias problem into a down-bias of its own, not toward balance
- Reframes the earlier equity curve finding: gb_recall beating gb_accuracy
  on Sharpe wasn't evidence that "being more cautious about up calls helps."
  This config effectively sits in cash most of the time (since it predicts
  down most of the time), which happened to avoid enough bad days to edge
  out gb_accuracy, despite not being well calibrated to the true ~55/45
  class split
- lr_accuracy and hgb_accuracy remain up-biased (catching only 20.7% and
  30.6% of actual down days respectively). rf_accuracy sits closest to
  the true base rate of the four, mildly down-leaning
- Takeaway: none of the 4 winning configs are well calibrated to actual
  class balance, each is biased in one direction or another, just to
  different degrees and different directions

### Next
- ROC curves and feature importance still planned for the 4 winning
  configs, per original Notebook 03 scope

## Notebook 03: ROC Curves (4 Winning Configs)

### Approach
- Modified generate_oos_predictions() to also collect predict_proba
  outputs (probability of class 1/"up"), added as a 'probability' column
- Reran the 4 winning configs through the updated function
- Single overlaid ROC plot, all 4 models plus diagonal random-guess
  reference, AUC in legend, saved to results/roc_curves_winning_configs.png

### Results
| Config       | AUC   |
|--------------|-------|
| lr_accuracy  | 0.484 |
| rf_accuracy  | 0.507 |
| gb_recall    | 0.507 |
| hgb_accuracy | 0.498 |

### Key findings
- All four AUCs sit in a tight band around 0.5 (coin flip), every curve
  hugs the diagonal, none of the models can meaningfully rank days by
  likelihood of an up move
- LR's 0.484 being below 0.5 is noise around the coin-flip line, not
  anti-skill
- Most important: hgb_accuracy had the best equity curve of all 8 configs
  (Sharpe 0.788) but an AUC of 0.498. Its backtest performance came from
  its up-bias aligning with the market's upward drift, not from predictive
  skill. ROC is what separates "profitable because biased in a lucky
  direction" from "profitable because skilled"
- Three independent lines of evidence now converge on the same conclusion:
  no config beats baseline accuracy, no config beats buy-and-hold, no
  config ranks days better than chance. Daily SPY direction resists
  prediction with technical indicators alone, this is the project's
  honest core finding


### Metrics Bar Chart
- Combined accuracy, recall_down, Sharpe, and AUC for the 4 winning configs
  into one grouped bar chart with reference lines (0.550 baseline accuracy,
  0.875 buy-and-hold Sharpe, 0.500 coin-flip AUC), saved to
  results/metrics_comparison_winning_configs.png
- Visual confirms the convergent finding: every bar falls short of its
  relevant reference line
- Notable contrast: recall_down varies widely across configs (0.207 to
  0.658) while AUC barely varies at all (~0.5 for everyone), different
  bias choices, identical underlying skill

### Feature Importance (RF, GB) + LR Coefficients
- Refit each winning config once on the full dataset purely to extract
  importances (walk-forward fold models aren't retained by the function);
  these describe what models leaned on, not validated predictive signal
- Tree importances are notably flat (RF ~0.05-0.095, GB ~0.04-0.11), no
  dominant feature in either model, consistent with the coin-flip AUC
  finding: attention spread evenly because nothing helps much
- OBV, Volume, Daily_Return rank high in both tree models, the features
  Notebook 01 flagged as most independent/non-redundant
- LR quirk: Daily_Return has the largest negative coefficient
  (mean-reversion lean) while ROC has the largest positive one (momentum
  lean), internally inconsistent, consistent with fitting noise
- HGB excluded (no native feature_importances_), LR coefficients included
  as a third panel with the caveat that coefficients measure direction +
  influence on scaled features, not the same quantity as tree importances
- Saved to results/feature_importance_winning_configs.png

### Model Finalization: v2
- hgb_accuracy retrained on the full dataset and saved as models/hgb_v2.joblib,
  superseding hgb_v1.joblib as the selected model for Project 3
- Hyperparameters: class_weight=None, learning_rate=0.01, max_depth=3,
  max_iter=100, random_state=42
- No scaler saved this time (unlike v1): HGB is tree-based, tuned and
  evaluated unscaled, so Project 3's predictor loads only the model file
- Same finalization logic as v1: full-data retrain happens after all
  evaluation, never re-reported as a validation metric
- Selection caveat, documented deliberately: v2 is the best of a set of
  models with no demonstrated predictive skill (AUC 0.498). Its equity curve
  edge comes from up-bias aligning with market drift. Deployed in Project 3
  to demonstrate deployment engineering, not model quality

### metrics.json Updated
- selected_model replaced: now hgb_accuracy / hgb_v2.joblib with full
  hyperparameters, selection criterion (equity curve Sharpe and total
  return across 8 tuned configs), and selection date 2026-07-07
- Added tuned_config_comparison: accuracy, recall_down, AUC, Sharpe, total
  return, max drawdown for all 8 configs
- Added buy_and_hold benchmark block (Sharpe 0.875, total return 2.965,
  max drawdown -0.337)
- Added key_finding string with the honest caveat, travels with the model
  so Project 3's /model/info endpoint never overstates what it serves
- Notebook 02's original entries preserved (file appended to, not
  overwritten)

### Notebook 03 Hygiene
- Restart Kernel and Run All exposed two leftover issues: the 4-winner
  rerun cell placed above model_configs (ordering), and the original
  single-config sanity test cells referencing hgb_recall_results (obsolete
  scaffolding). Both resolved by deletion, rerun cell no longer needed
  since the final function version collects probabilities on every run
- Notebook now runs clean end-to-end from a cold kernel, meeting the
  "renders on a fresh clone" definition of done
- Lesson logged: every cell must sit below everything it references; when
  Run All throws a NameError, move the failing cell down rather than the
  definition up

### Project 2 Status
- COMPLETE except README (deliberately deferred until all results settled,
  which they now are)
- Notebook 04 dropped (documented separately), project ends at Notebook 03
- Next session: write the README, then begin Project 3 (FastAPI service
  wrapping hgb_v2.joblib + metrics.json)