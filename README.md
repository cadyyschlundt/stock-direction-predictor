# ML Direction Predictor

## About

Build a classifier predicting whether SPY closes up or down the next trading day using technical indicators from Project 1. Demonstrates proper time-series ML: walk-forward validation (no leakage), honest model comparison, and deployment readiness. Baseline to beat: 55% accuracy (always predict "up"). This is the second project of a larger project I am working on, with the goal being to improve my coding skills and understand important financial concepts and market dynamics.

## Learning Approach

This project was intentionally built as a learning experience, and as the second step of a larger project. I used AI, but with a specific constraint: it could only generate code if I explicitly asked for it. 

Instead, I had the AI guide me through concepts, explain tradeoffs, and walk me through what the next step should be before writing any code, and double checking that any code I had written was accurate. This meant I had to actively work through each part of the pipeline rather than just copying generated code.

I also supplemented this with AWS courses, sklearn documentation, and financial data resources. Every major decision (walk-forward validation, model selection criterion, feature engineering tradeoffs) came from reasoning through the problem first, then implementing it.

This approach meant slower initial progress, but it forced me to actually understand what the code is doing and why it matters. The honest finding that no model beats baseline came directly from this approach, not from lack of effort. 

## AI Usage
I was curious on what it would look like to try to beat the baseline. I had the claude code feature attempt to restart the project with a goal of beating the baseline by at least one percent, and pushed it further a few times. It explored a more aggressive hyperparameter search, and while it did bea the baseline it was all overfitting. I am going forward with the finding I implemented as using that in the following projects.

See `attempt_2/` folder for full exploration.

## Architecture
Data (SPY 2014-2026, 2,939 rows)
↓
15 Features (technical indicators, redundant price-level columns dropped)
↓
Walk-Forward Validation (5 time-ordered folds)
↓
4 Model Families × 40+ hyperparameter combinations
↓
Selected: HistGradientBoosting (best equity curve Sharpe)
↓
models/hgb_v2.joblib + results/metrics.json

## Results

**Before tuning (initial baseline models):**

| Model | Accuracy | Sharpe |
|-------|----------|--------|
| Logistic Regression | 51.3% | 0.564 |
| Random Forest | 49.2% | 0.564 |
| Gradient Boosting | 48.4% | 0.564 |
| HistGradientBoosting | 50.0% | 0.564 |
| Baseline (always "up") | 55.0% | 0.877 |

**After tuning (40+ hyperparameter combinations):**

| Model | Accuracy | AUC |
|-------|----------|-----|
| Logistic Regression | 52.8% | 0.484 |
| Random Forest | 49.5% | 0.507 |
| Gradient Boosting | 49.3% | 0.507 |
| HistGradientBoosting | 52.6% | 0.498 |
| Baseline (always "up") | 55.0% | 0.500 |

**Selected model (HistGradientBoosting) equity curve:**

| Metric | Model | Buy & Hold |
|--------|-------|-----------|
| Sharpe | 0.788 | 0.875 |
| Total Return | 215.4% | 296.5% |
| Max Drawdown | -28.3% | -33.7% |

## What This Means

Best accuracy after tuning was 52.8% (Logistic Regression), still below the 55% baseline. HistGradientBoosting had 52.6% accuracy but the best equity curve Sharpe (0.788 vs LR's 0.725). 

The AUC for all models stayed around 0.5 (coin flip), so none of them are actually predicting direction better. HGB just predicts "up" 70% of the time, which happens to align with the market's upward drift. The model's profit comes from bias, not skill.

## How I Built It

**Features:** 15 final indicators (dropped 9 price-level columns that measured similar things)

**Models:** Logistic Regression, Random Forest, Gradient Boosting, HistGradientBoosting 

**Tuning:** GridSearchCV with equity curve Sharpe as the selection criterion. Tested 40+ combinations across four model families.

**Validation:** TimeSeriesSplit (5 folds, no leakage, no future peeking)

## Files
stock-direction-predictor/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 02b_tuning.ipynb
│   ├── 03_model_comparison.ipynb
├── src/
│   ├── init.py
│   ├── data.py
│   ├── predictor.py
│   └── models_info.py
├── tests/
│   ├── test_data.py
│   ├── test_predictor.py
│   └── test_models_info.py
├── models/
│   └── hgb_v2.joblib
├── results/
│   ├── metrics.json
│   ├── equity_curve.png
│   ├── confusion_matrix_lr_accuracy.png
│   ├── confusion_matrix_rf_accuracy.png
│   ├── confusion_matrix_gb_recall.png
│   ├── confusion_matrix_hgb_accuracy.png
│   ├── roc_curves_winning_configs.png
│   ├── feature_importance_winning_configs.png
│   └── metrics_comparison_winning_configs.png
└── attempt_2/
└── (exploration of beating baseline through overfitting)
## How to Run

```bash
pip install -r requirements.txt
pytest tests/  # All tests pass
```

Load the model:
```python
from src.predictor import StockDirectionPredictor
from src.data import load_and_prepare_data

X, y, df = load_and_prepare_data()
predictor = StockDirectionPredictor()
predictions = predictor.predict(X)
probabilities = predictor.predict_proba(X)
```

## What I Learned

- Walk-forward validation is non-negotiable for time series
- Honest findings are more credible than inflated claims
- AUC catches the difference between "skilled" and "lucky bias"
- Equity curve Sharpe is better than raw accuracy for trading
- Feature redundancy kills signal; the 9 price columns were noise
- how to build ML models

## Next

Project 3 (FastAPI service wrapping hgb_v2.joblib)
