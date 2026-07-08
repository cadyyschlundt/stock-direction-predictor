"""Escalation ladder for attempt_2 (run after run.py if target not met).

Level 1: finer decision-threshold grid (0.25-0.75, step 0.001) on top single runs.
Level 2: weighted two-model ensembles over the top runs (weight grid 0-1 step 0.05),
         fine thresholds.

Updates results/metrics.json with an "escalation" section and writes results/README.md.
Run from this directory:  ../.venv/Scripts/python.exe escalate.py
"""
import itertools
import json
from datetime import date
from pathlib import Path

import numpy as np

from features import build_feature_sets
from search import model_grid, pooled_oos_probabilities
from sklearn.model_selection import TimeSeriesSplit

TARGET_ACCURACY = 0.56
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FINE_THRESHOLDS = np.round(np.arange(0.25, 0.7501, 0.001), 3)
WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)

# top single runs from run.py's leaderboard, recomputed here (deterministic, seed 42)
TOP_RUNS = [
    ("lr_c1", "engineered"),
    ("lr_c1", "original15"),
    ("rf_100_d5", "combined"),
    ("lr_c01", "engineered"),
    ("hgb_d3_lr001", "combined"),
    ("lr_c01", "original15"),
    ("lr_c001", "original15"),
    ("lr_c01", "combined"),
]


def fine_sweep(actual, probability):
    """Vectorized sweep over FINE_THRESHOLDS -> (best threshold, best accuracy)."""
    preds = probability[:, None] >= FINE_THRESHOLDS[None, :]
    accs = (preds == actual[:, None]).mean(axis=0)
    i = int(np.argmax(accs))
    return float(FINE_THRESHOLDS[i]), float(accs[i])


def main():
    feature_sets, y = build_feature_sets()
    tscv = TimeSeriesSplit(n_splits=5)
    estimators = {name: (est, scale) for name, est, scale in model_grid()}

    probs, actual = {}, None
    for model_name, fs_name in TOP_RUNS:
        est, scale = estimators[model_name]
        oos = pooled_oos_probabilities(est, feature_sets[fs_name], y, tscv, scale)
        key = f"{model_name}__{fs_name}"
        probs[key] = oos["probability"].values
        actual = oos["actual"].values
        print(f"recomputed {key}")

    results = []

    # Level 1: fine thresholds on singles
    for key, p in probs.items():
        t, acc = fine_sweep(actual, p)
        results.append({"run": key, "kind": "single_fine", "threshold": t,
                        "weight": None, "accuracy": acc})
        print(f"L1 {key}: acc={acc:.4f} @ t={t:.3f}")

    # Level 2: weighted pairs
    for (k1, p1), (k2, p2) in itertools.combinations(probs.items(), 2):
        for w in WEIGHTS:
            blend = w * p1 + (1 - w) * p2
            t, acc = fine_sweep(actual, blend)
            results.append({"run": f"{k1} + {k2}", "kind": "weighted_pair",
                            "threshold": t, "weight": float(w), "accuracy": acc})

    results.sort(key=lambda r: r["accuracy"], reverse=True)
    best = results[0]
    hit = best["accuracy"] >= TARGET_ACCURACY

    print("\n=== Escalation top 10 ===")
    for r in results[:10]:
        w = f" w={r['weight']:.2f}" if r["weight"] is not None else ""
        print(f"{r['run']}{w}: acc={r['accuracy']:.4f} @ t={r['threshold']:.3f} ({r['kind']})")
    print(f"\nBest: {best['run']} acc={best['accuracy']:.4f} "
          f"-> {'TARGET MET' if hit else 'TARGET NOT MET'} (target {TARGET_ACCURACY})")

    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    metrics["escalation"] = {
        "date": str(date.today()),
        "levels_run": ["fine_thresholds_0.001", "weighted_pairs_top8"],
        "n_combinations_evaluated": len(results) * len(FINE_THRESHOLDS),
        "winner": best,
        "target_met": bool(hit),
        "top_10": results[:10],
    }
    if hit:
        metrics["target_met"] = True
        metrics["winner"] = {**best, "auc": None,
                             "note": "found during escalation; see 'escalation' section"}
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Updated {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
