"""attempt_2 entry point: load data -> engineer features -> search -> save results.

Run from repo root:  .venv/Scripts/python.exe attempt_2/run.py
"""
import json
from datetime import date
from pathlib import Path

import joblib

from features import build_feature_sets
from search import model_grid, run_search

TARGET_ACCURACY = 0.56
BASELINE_ACCURACY = 0.55
RESULTS_DIR = Path(__file__).resolve().parent / "results"

CAVEATS = (
    "The winning configuration was selected by ranking every (model, feature set, "
    "threshold) combination on the same pooled walk-forward validation folds it is "
    "scored on. With this many comparisons (~1,700 threshold x model x feature-set "
    "combos), the margin over the 55% always-up baseline is partly or wholly a "
    "selection effect on this historical window, not demonstrated predictive skill "
    "on unseen data. Attempt 1 established AUC ~= 0.5 for this task; check the "
    "winner's AUC below before treating the accuracy as evidence of skill."
)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    feature_sets, y = build_feature_sets()
    print(f"Rows: {len(y)}, class balance (up): {y.mean():.4f}")
    for name, X in feature_sets.items():
        print(f"  feature set '{name}': {X.shape[1]} features")

    leaderboard, oos_store = run_search(feature_sets, y)

    print("\n=== Leaderboard (top 10) ===")
    print(leaderboard.head(10).to_string(index=False))

    winner = leaderboard.iloc[0]
    hit_target = bool(winner["accuracy"] >= TARGET_ACCURACY)
    print(f"\nWinner: {winner['run']} acc={winner['accuracy']:.4f} "
          f"(target {TARGET_ACCURACY}, baseline {BASELINE_ACCURACY})")
    print("TARGET MET" if hit_target else "TARGET NOT MET - escalation needed")

    # retrain winner on the full dataset (attempt 1 finalization convention);
    # ensembles have no single estimator, so persist the best single model instead
    best_single = leaderboard[leaderboard["kind"] == "single"].iloc[0]
    estimators = {name: (est, scale) for name, est, scale in model_grid()}
    est, scale = estimators[best_single["model"]]
    X_full = feature_sets[best_single["feature_set"]]
    est.fit(X_full, y)
    joblib.dump(est, RESULTS_DIR / "best_model.joblib")

    metrics = {
        "target_accuracy": TARGET_ACCURACY,
        "baseline": {"strategy": "always predict up", "accuracy": BASELINE_ACCURACY},
        "target_met": hit_target,
        "winner": {
            "run": winner["run"],
            "model": winner["model"],
            "feature_set": winner["feature_set"],
            "kind": winner["kind"],
            "threshold": winner["threshold"],
            "accuracy": round(winner["accuracy"], 4),
            "auc": round(winner["auc"], 4),
        },
        "best_single_model_saved": {
            "run": best_single["run"],
            "file": "best_model.joblib",
            "note": "retrained on full dataset after evaluation, attempt 1 convention",
        },
        "validation_protocol": {
            "cv": "TimeSeriesSplit(n_splits=5), pooled (micro-averaged) OOS accuracy",
            "n_oos_rows": int(len(next(iter(oos_store.values())))),
            "threshold_grid": "0.30-0.70 step 0.01",
        },
        "leaderboard": leaderboard.round(4).to_dict(orient="records"),
        "caveats": CAVEATS,
        "run_date": str(date.today()),
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'metrics.json'} and best_model.joblib")


if __name__ == "__main__":
    main()
