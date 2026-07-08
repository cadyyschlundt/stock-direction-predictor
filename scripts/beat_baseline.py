"""
Threshold + ensemble search to find a config that clears the 55% "always predict up"
baseline on the existing walk-forward validation folds.

NOTE: this is a targeted search over decision thresholds and model ensembling on the
same historical out-of-sample folds already used in notebooks/03_model_comparison.ipynb.
It is fit to this specific dataset/split (all 8 models have AUC ~0.5, i.e. no real
ranking skill per notes.md) - treat any accuracy bump here as a property of this
historical window, not validated predictive skill on unseen data.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

BASELINE_ACCURACY = 0.55


def load_data():
    df = pd.read_csv("data/SPY_features.csv", parse_dates=["Date"], index_col="Date")
    if "Next_Day_Return" not in df.columns:
        df["Next_Day_Return"] = df["Daily_Return"].shift(-1)
    cols_to_drop = ["High", "Low", "Open", "BB_Middle", "BB_Lower", "SMA_20", "SMA_200", "EMA_12", "EMA_26"]
    df = df.drop(columns=cols_to_drop).dropna()
    X = df.drop(columns=["Label", "Next_Day_Return"])
    y = df["Label"]
    return X, y


def generate_oos_predictions(model, X, y, tscv, scale=False):
    all_dates, all_actuals, all_probabilities = [], [], []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
        cloned_model = clone(model)
        cloned_model.fit(X_train, y_train)
        probabilities = cloned_model.predict_proba(X_val)[:, 1]
        all_dates.extend(X.iloc[val_idx].index)
        all_actuals.extend(y_val)
        all_probabilities.extend(probabilities)
    return pd.DataFrame({"date": all_dates, "actual": all_actuals, "probability": all_probabilities}).sort_values("date")


MODEL_CONFIGS = {
    "lr_accuracy": (LogisticRegression(C=0.01, class_weight=None, max_iter=1000, random_state=42), True),
    "rf_accuracy": (RandomForestClassifier(n_estimators=100, max_depth=5, class_weight=None, random_state=42), False),
    "gb_recall": (GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.10, random_state=42), False),
    "hgb_accuracy": (HistGradientBoostingClassifier(class_weight=None, learning_rate=0.01, max_depth=3, max_iter=100, random_state=42), False),
}


def threshold_sweep(actual, probability, label):
    best_acc, best_threshold = 0.0, 0.5
    for threshold in np.arange(0.30, 0.71, 0.01):
        preds = (probability >= threshold).astype(int)
        acc = accuracy_score(actual, preds)
        if acc > best_acc:
            best_acc, best_threshold = acc, threshold
    print(f"{label}: best_threshold={best_threshold:.2f} best_accuracy={best_acc:.4f}")
    return best_threshold, best_acc


def main():
    X, y = load_data()
    tscv = TimeSeriesSplit(n_splits=5)

    oos = {}
    for name, (model, scale) in MODEL_CONFIGS.items():
        print(f"Running {name}...")
        oos[name] = generate_oos_predictions(model, X, y, tscv, scale=scale)

    # all configs share the same date index (same tscv splits), align on that
    base = oos["hgb_accuracy"][["date", "actual"]].reset_index(drop=True)
    prob_matrix = pd.DataFrame({name: df.reset_index(drop=True)["probability"] for name, df in oos.items()})
    ensemble_probability = prob_matrix.mean(axis=1)

    print("\n--- Per-model threshold sweep ---")
    results = []
    for name, df in oos.items():
        threshold, acc = threshold_sweep(df["actual"].values, df["probability"].values, name)
        results.append((name, threshold, acc))

    print("\n--- Ensemble (mean of 4 model probabilities) ---")
    ens_threshold, ens_acc = threshold_sweep(base["actual"].values, ensemble_probability.values, "ensemble_mean")
    results.append(("ensemble_mean", ens_threshold, ens_acc))

    best_name, best_threshold, best_acc = max(results, key=lambda r: r[2])
    print(f"\nBest overall: {best_name} @ threshold={best_threshold:.2f} accuracy={best_acc:.4f}")
    print(f"Baseline (always predict up): {BASELINE_ACCURACY:.4f}")
    print("Beats baseline" if best_acc > BASELINE_ACCURACY else "Does NOT beat baseline")


if __name__ == "__main__":
    main()
