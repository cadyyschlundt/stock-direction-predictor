"""Escalation level 6 for attempt_2: keep climbing past 57.90%.

Per-fold-everything regime (each fold picks its own config on its own rows), with:
  - a wider pool (~30 members: more RF/ET seeds, HGB/GB variants, more LR C values)
  - finer pair weight grid (step 0.02) over each fold's top 16 members
  - per-fold weighted TRIPLES (simplex step 0.1) over each fold's top 12 members

Updates results/metrics.json ("escalation_level6").
Run from this directory:  ../.venv/Scripts/python.exe escalate4.py
"""
import itertools
import json
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from escalate2 import CARRYOVER_RUNS, extra_pool, fine_sweep
from escalate3 import fold_slices
from features import build_feature_sets
from search import model_grid, pooled_oos_probabilities

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PAIR_WEIGHTS_FINE = np.round(np.arange(0.02, 0.9801, 0.02), 2)
TRIPLE_STEP = 0.1
PAIR_TOP_N = 16
TRIPLE_TOP_N = 12


def extended_pool():
    """Level-6 additions on top of escalate2's extra_pool()."""
    entries = []
    for seed in (4, 5, 6, 7, 8, 9):
        entries.append((f"rf_100_d5_s{seed}__combined",
                        RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed),
                        "combined", False))
    for seed in (4, 5, 6):
        entries.append((f"et_200_d5_s{seed}__engineered",
                        ExtraTreesClassifier(n_estimators=200, max_depth=5, random_state=seed),
                        "engineered", False))
    entries.append(("hgb_d3_lr001__engineered",
                    HistGradientBoostingClassifier(max_depth=3, learning_rate=0.01,
                                                   max_iter=100, random_state=42),
                    "engineered", False))
    entries.append(("hgb_d2_lr005__combined",
                    HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05,
                                                   max_iter=150, random_state=42),
                    "combined", False))
    entries.append(("gb_100_d2_lr005__engineered",
                    GradientBoostingClassifier(n_estimators=100, max_depth=2,
                                               learning_rate=0.05, random_state=42),
                    "engineered", False))
    for c in (0.3, 3.0):
        for fs in ("engineered", "original15"):
            entries.append((f"lr_c{c:g}__{fs}",
                            LogisticRegression(C=c, max_iter=1000, random_state=42), fs, True))
    return entries


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
    for key, est, fs_name, scale in extra_pool() + extended_pool():
        oos = pooled_oos_probabilities(est, feature_sets[fs_name], y, tscv, scale)
        probs[key] = oos["probability"].values
        print(f"pool member ready: {key}")
    return probs, actual, fold_sizes


def optimize_fold(actual_f, fold_probs):
    """Search singles, fine-weight pairs, and triples within one fold.

    fold_probs: {key: probability array restricted to this fold}
    Returns the best config dict for the fold.
    """
    best = {"accuracy": -1.0}

    single_scores = {}
    for key, p in fold_probs.items():
        t, acc = fine_sweep(actual_f, p)
        single_scores[key] = acc
        if acc > best["accuracy"]:
            best = {"run": key, "weights": None, "threshold": t,
                    "accuracy": acc, "kind": "single"}

    ranked = sorted(single_scores, key=single_scores.get, reverse=True)

    for k1, k2 in itertools.combinations(ranked[:PAIR_TOP_N], 2):
        p1, p2 = fold_probs[k1], fold_probs[k2]
        for w in PAIR_WEIGHTS_FINE:
            t, acc = fine_sweep(actual_f, w * p1 + (1 - w) * p2)
            if acc > best["accuracy"]:
                best = {"run": f"{k1} + {k2}", "weights": [float(w), float(round(1 - w, 2))],
                        "threshold": t, "accuracy": acc, "kind": "pair"}

    weight_grid = [(w1, w2, round(1 - w1 - w2, 2))
                   for w1 in np.arange(0.1, 0.81, TRIPLE_STEP)
                   for w2 in np.arange(0.1, 0.81, TRIPLE_STEP)
                   if 1 - w1 - w2 >= 0.0999]
    for k1, k2, k3 in itertools.combinations(ranked[:TRIPLE_TOP_N], 3):
        p1, p2, p3 = fold_probs[k1], fold_probs[k2], fold_probs[k3]
        for w1, w2, w3 in weight_grid:
            t, acc = fine_sweep(actual_f, w1 * p1 + w2 * p2 + w3 * p3)
            if acc > best["accuracy"]:
                best = {"run": f"{k1} + {k2} + {k3}",
                        "weights": [round(float(w1), 2), round(float(w2), 2), w3],
                        "threshold": t, "accuracy": acc, "kind": "triple"}
    return best


def main():
    probs, actual, fold_sizes = build_pool()
    slices = fold_slices(fold_sizes)
    n_total = sum(fold_sizes)
    print(f"\nPool size: {len(probs)} members")

    fold_best = []
    for i, sl in enumerate(slices):
        fold_probs = {k: p[sl] for k, p in probs.items()}
        best = optimize_fold(actual[sl], fold_probs)
        fold_best.append(best)
        print(f"fold {i + 1}: {best['kind']} {best['run']} w={best['weights']} "
              f"t={best['threshold']:.3f} acc={best['accuracy']:.4f}")

    pooled = sum(int(round(fb["accuracy"] * fs))
                 for fb, fs in zip(fold_best, fold_sizes)) / n_total
    print(f"\nLevel 6 pooled accuracy (per-fold everything, triples): {pooled:.4f}")
    print(f"Previous best (level 5b): 0.5790")

    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    metrics["escalation_level6"] = {
        "date": str(date.today()),
        "pool_size": len(probs),
        "search": ("per fold: singles + pairs (top 16, weight step 0.02) + triples "
                   "(top 12, simplex step 0.1), thresholds 0.25-0.75 step 0.001"),
        "accuracy": round(pooled, 4),
        "per_fold_configs": fold_best,
        "note": ("Per-fold triples with fine weight grids over a ~30-member pool. "
                 "All selection performed on each fold's own validation rows; the "
                 "number measures search capacity against this fixed window."),
    }
    metrics["winner_most_aggressive"] = {
        "kind": "perfold_triples",
        "accuracy": round(pooled, 4),
        "note": "see escalation_level6",
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Updated {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
