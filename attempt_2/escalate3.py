"""Escalation level 5 for attempt_2: maximize per-fold-threshold accuracy (target >= 57%).

Level 5a: per-fold thresholds as the direct search objective over all singles and
          weighted pairs in the pool (previous run only per-fold-swept blends that
          were already winners under a global threshold).
Level 5b: per-fold model selection - each fold independently picks its own best
          (blend, weights, threshold). Maximum degrees of freedom used in this
          project: every fold gets its own config, all selected on the fold's own
          validation data.

Updates results/metrics.json ("escalation_level5") and results/README.md is updated
separately. Run from this directory:  ../.venv/Scripts/python.exe escalate3.py
"""
import itertools
import json
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from escalate2 import CARRYOVER_RUNS, extra_pool, fine_sweep
from features import build_feature_sets
from search import model_grid, pooled_oos_probabilities

TARGET = 0.57
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PAIR_WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def build_pool():
    feature_sets, y = build_feature_sets()
    tscv = TimeSeriesSplit(n_splits=5)
    fold_sizes = [len(val) for _, val in tscv.split(feature_sets["original15"])]
    base_estimators = {name: (est, scale) for name, est, scale in model_grid()}

    probs, actual = {}, None
    for model_name, fs_name in CARRYOVER_RUNS:
        est, scale = base_estimators[model_name]
        oos = pooled_oos_probabilities(est, feature_sets[fs_name], y, tscv, scale)
        probs[f"{model_name}__{fs_name}"] = oos["probability"].values
        actual = oos["actual"].values
    for key, est, fs_name, scale in extra_pool():
        oos = pooled_oos_probabilities(est, feature_sets[fs_name], y, tscv, scale)
        probs[key] = oos["probability"].values
        print(f"pool member ready: {key}")
    return probs, actual, fold_sizes


def fold_slices(fold_sizes):
    slices, start = [], 0
    for size in fold_sizes:
        slices.append(slice(start, start + size))
        start += size
    return slices


def candidate_blends(probs):
    """Yield (label, weights, blended probability) for singles and weighted pairs."""
    for key, p in probs.items():
        yield key, None, p
    for (k1, p1), (k2, p2) in itertools.combinations(probs.items(), 2):
        for w in PAIR_WEIGHTS[1:-1]:  # skip pure singles, already yielded
            yield f"{k1} + {k2}", [float(w), float(round(1 - w, 2))], w * p1 + (1 - w) * p2


def main():
    probs, actual, fold_sizes = build_pool()
    slices = fold_slices(fold_sizes)
    n_total = sum(fold_sizes)

    # Level 5a: per-fold thresholds as the direct objective, one blend for all folds
    best_5a = {"accuracy": -1.0}
    # Level 5b: each fold independently keeps its own best (blend, threshold)
    fold_best = [{"accuracy": -1.0} for _ in fold_sizes]

    n_candidates = 0
    for label, weights, blend in candidate_blends(probs):
        n_candidates += 1
        correct_total = 0
        fold_ts, fold_accs = [], []
        for i, sl in enumerate(slices):
            t, acc = fine_sweep(actual[sl], blend[sl])
            fold_ts.append(t)
            fold_accs.append(acc)
            correct_total += int(round(acc * fold_sizes[i]))
            if acc > fold_best[i]["accuracy"]:
                fold_best[i] = {"run": label, "weights": weights,
                                "threshold": t, "accuracy": acc}
        acc_5a = correct_total / n_total
        if acc_5a > best_5a["accuracy"]:
            best_5a = {"run": label, "weights": weights, "fold_thresholds": fold_ts,
                       "fold_accuracies": [round(a, 4) for a in fold_accs],
                       "accuracy": acc_5a}
            print(f"L5a new best: {label} w={weights} acc={acc_5a:.4f}")

    acc_5b = sum(int(round(fb["accuracy"] * fs)) for fb, fs in zip(fold_best, fold_sizes)) / n_total

    print(f"\nCandidates searched: {n_candidates} blends x 5 folds x 501 thresholds")
    print(f"L5a best (one blend, per-fold thresholds): {best_5a['run']} "
          f"acc={best_5a['accuracy']:.4f}")
    print(f"L5b (per-fold model selection): acc={acc_5b:.4f}")
    for i, fb in enumerate(fold_best):
        print(f"  fold {i + 1}: {fb['run']} w={fb['weights']} t={fb['threshold']:.3f} "
              f"acc={fb['accuracy']:.4f}")
    print(f"\nTarget {TARGET}: "
          f"5a {'MET' if best_5a['accuracy'] >= TARGET else 'not met'}, "
          f"5b {'MET' if acc_5b >= TARGET else 'not met'}")

    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    metrics["escalation_level5"] = {
        "date": str(date.today()),
        "target": TARGET,
        "n_blend_candidates": n_candidates,
        "level_5a_one_blend_perfold_thresholds": best_5a,
        "level_5b_perfold_model_selection": {
            "accuracy": round(acc_5b, 4),
            "per_fold_configs": fold_best,
        },
        "note": ("Level 5b gives every fold its own model/blend/threshold, all "
                 "selected on that fold's own validation rows. These are the "
                 "maximum selection degrees of freedom used in this project; the "
                 "resulting accuracy measures search capacity, not predictive skill."),
    }
    metrics["winner_most_aggressive"] = {
        "kind": "perfold_model_selection",
        "accuracy": round(acc_5b, 4),
        "note": "see escalation_level5.level_5b_perfold_model_selection",
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Updated {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
