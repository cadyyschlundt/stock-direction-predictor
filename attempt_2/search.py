"""Walk-forward search for attempt_2.

Protocol (identical to attempt 1 for comparability):
  - TimeSeriesSplit(n_splits=5), scaler fit on train folds only (LR only)
  - out-of-sample predictions pooled ("stitched") across all 5 folds
  - accuracy computed once over the pooled set (micro-average)

Search space: model grid x feature sets x decision thresholds, plus
mean-probability ensembles of the top single models.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
THRESHOLDS = np.round(np.arange(0.30, 0.7001, 0.01), 2)


def model_grid():
    """(name, estimator, needs_scaling) triples."""
    return [
        ("lr_c001", LogisticRegression(C=0.01, max_iter=1000, random_state=RANDOM_STATE), True),
        ("lr_c01", LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE), True),
        ("lr_c1", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE), True),
        ("rf_100_d5", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE), False),
        ("rf_200_d3", RandomForestClassifier(n_estimators=200, max_depth=3, random_state=RANDOM_STATE), False),
        ("rf_200_dNone", RandomForestClassifier(n_estimators=200, max_depth=None, random_state=RANDOM_STATE), False),
        ("et_200_d5", ExtraTreesClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE), False),
        ("et_200_dNone", ExtraTreesClassifier(n_estimators=200, max_depth=None, random_state=RANDOM_STATE), False),
        ("gb_100_d3_lr01", GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE), False),
        ("gb_100_d2_lr005", GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=RANDOM_STATE), False),
        ("hgb_d3_lr001", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.01, max_iter=100, random_state=RANDOM_STATE), False),
        ("hgb_d3_lr005", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=RANDOM_STATE), False),
        ("hgb_d2_lr01", HistGradientBoostingClassifier(max_depth=2, learning_rate=0.1, max_iter=100, random_state=RANDOM_STATE), False),
    ]


def pooled_oos_probabilities(model, X, y, tscv, scale):
    """Fit per fold, return pooled OOS (dates, actuals, P(up)) sorted by date."""
    dates, actuals, probs = [], [], []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
        fitted = clone(model).fit(X_train, y_train)
        probs.extend(fitted.predict_proba(X_val)[:, 1])
        dates.extend(X.index[val_idx])
        actuals.extend(y.iloc[val_idx])
    out = pd.DataFrame({"date": dates, "actual": actuals, "probability": probs})
    return out.sort_values("date").reset_index(drop=True)


def best_threshold(actual, probability):
    """Sweep THRESHOLDS, return (threshold, accuracy) with the highest accuracy."""
    best_t, best_acc = 0.5, -1.0
    for t in THRESHOLDS:
        acc = accuracy_score(actual, (probability >= t).astype(int))
        if acc > best_acc:
            best_t, best_acc = t, acc
    return float(best_t), float(best_acc)


def run_search(feature_sets, y, n_splits=5, top_k_ensemble=(2, 3, 4, 5)):
    """Return (leaderboard DataFrame, {run_key: pooled oos DataFrame})."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows, oos_store = [], {}

    for fs_name, X in feature_sets.items():
        for model_name, model, scale in model_grid():
            key = f"{model_name}__{fs_name}"
            oos = pooled_oos_probabilities(model, X, y, tscv, scale)
            oos_store[key] = oos
            t, acc = best_threshold(oos["actual"].values, oos["probability"].values)
            rows.append({
                "run": key, "model": model_name, "feature_set": fs_name,
                "kind": "single", "threshold": t, "accuracy": acc,
                "auc": roc_auc_score(oos["actual"], oos["probability"]),
            })
            print(f"{key}: acc={acc:.4f} @ t={t:.2f}")

    # mean-probability ensembles of the top single runs
    singles = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    ref = next(iter(oos_store.values()))
    for k in top_k_ensemble:
        members = singles.head(k)["run"].tolist()
        mean_prob = np.mean([oos_store[m]["probability"].values for m in members], axis=0)
        t, acc = best_threshold(ref["actual"].values, mean_prob)
        key = f"ensemble_top{k}"
        oos_store[key] = pd.DataFrame({
            "date": ref["date"], "actual": ref["actual"], "probability": mean_prob,
        })
        rows.append({
            "run": key, "model": "+".join(members), "feature_set": "mixed",
            "kind": "ensemble", "threshold": t, "accuracy": acc,
            "auc": roc_auc_score(ref["actual"], mean_prob),
        })
        print(f"{key} ({members}): acc={acc:.4f} @ t={t:.2f}")

    leaderboard = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    return leaderboard, oos_store
