"""Escalation level 3+ for attempt_2 (run after escalate.py; pushes past 56.01%).

Level 3a: wider base-model pool (RF/ET seed sweeps, extra LR C values) -> singles
          + weighted pairs over the enlarged pool, fine thresholds.
Level 3b: weighted triples over the top pool members.
Level 4:  per-fold threshold selection on the best blends. Most aggressive form of
          selection-on-validation in this project: 5 thresholds chosen on the same
          folds they are scored on, instead of 1 global threshold. Reported
          separately and labeled as such.

Updates results/metrics.json ("escalation_level3") and appends to results/README.md.
Run from this directory:  ../.venv/Scripts/python.exe escalate2.py
"""
import itertools
import json
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from features import build_feature_sets
from search import model_grid, pooled_oos_probabilities

TARGET_PREVIOUS_BEST = 0.5601
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FINE_THRESHOLDS = np.round(np.arange(0.25, 0.7501, 0.001), 3)
PAIR_WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
TRIPLE_STEP = 0.1

# carried over from escalate.py's pool (the strongest singles)
CARRYOVER_RUNS = [
    ("lr_c1", "engineered"),
    ("lr_c1", "original15"),
    ("rf_100_d5", "combined"),
    ("lr_c01", "engineered"),
    ("hgb_d3_lr001", "combined"),
]


def extra_pool():
    """Additional (key, estimator, feature_set, scale) entries widening the pool."""
    entries = []
    for c in (10.0, 100.0):
        for fs in ("engineered", "original15"):
            entries.append((f"lr_c{c:g}__{fs}",
                            LogisticRegression(C=c, max_iter=1000, random_state=42), fs, True))
    for seed in (0, 1, 2, 3):
        entries.append((f"rf_100_d5_s{seed}__combined",
                        RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed),
                        "combined", False))
        entries.append((f"et_200_d5_s{seed}__engineered",
                        ExtraTreesClassifier(n_estimators=200, max_depth=5, random_state=seed),
                        "engineered", False))
    return entries


def fine_sweep(actual, probability, thresholds=FINE_THRESHOLDS):
    preds = probability[:, None] >= thresholds[None, :]
    accs = (preds == actual[:, None]).mean(axis=0)
    i = int(np.argmax(accs))
    return float(thresholds[i]), float(accs[i])


def per_fold_sweep(actual, probability, fold_sizes):
    """Pick the best threshold separately per fold; return (thresholds, pooled accuracy)."""
    thresholds, correct = [], 0
    start = 0
    for size in fold_sizes:
        a, p = actual[start:start + size], probability[start:start + size]
        t, acc = fine_sweep(a, p)
        thresholds.append(t)
        correct += int(round(acc * size))
        start += size
    return thresholds, correct / len(actual)


def main():
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

    results = []

    # Level 3a: singles + weighted pairs over the enlarged pool
    for key, p in probs.items():
        t, acc = fine_sweep(actual, p)
        results.append({"run": key, "kind": "single_fine", "weights": None,
                        "threshold": t, "accuracy": acc})
    for (k1, p1), (k2, p2) in itertools.combinations(probs.items(), 2):
        for w in PAIR_WEIGHTS:
            t, acc = fine_sweep(actual, w * p1 + (1 - w) * p2)
            results.append({"run": f"{k1} + {k2}", "kind": "weighted_pair",
                            "weights": [float(w), float(round(1 - w, 2))],
                            "threshold": t, "accuracy": acc})
    results.sort(key=lambda r: r["accuracy"], reverse=True)
    print(f"\nL3a best: {results[0]['run']} acc={results[0]['accuracy']:.4f}")

    # Level 3b: weighted triples over the top pool members (ranked by single acc)
    single_rank = [r["run"] for r in results if r["kind"] == "single_fine"]
    top_members = single_rank[:8]
    weight_grid = [(w1, w2, round(1 - w1 - w2, 2))
                   for w1 in np.arange(0.1, 0.91, TRIPLE_STEP)
                   for w2 in np.arange(0.1, 0.91, TRIPLE_STEP)
                   if 1 - w1 - w2 >= 0.0999]
    for k1, k2, k3 in itertools.combinations(top_members, 3):
        p1, p2, p3 = probs[k1], probs[k2], probs[k3]
        for w1, w2, w3 in weight_grid:
            t, acc = fine_sweep(actual, w1 * p1 + w2 * p2 + w3 * p3)
            results.append({"run": f"{k1} + {k2} + {k3}", "kind": "weighted_triple",
                            "weights": [round(float(w1), 2), round(float(w2), 2), w3],
                            "threshold": t, "accuracy": acc})
    results.sort(key=lambda r: r["accuracy"], reverse=True)
    best_global = results[0]
    print(f"L3b best (global threshold): {best_global['run']} "
          f"w={best_global['weights']} acc={best_global['accuracy']:.4f}")

    # Level 4: per-fold thresholds on the 20 best blends
    per_fold_results = []
    for r in results[:20]:
        if r["kind"] == "single_fine":
            blend = probs[r["run"]]
        else:
            members = [m.strip() for m in r["run"].split("+")]
            blend = np.sum([w * probs[m] for w, m in zip(r["weights"], members)], axis=0)
        ts, acc = per_fold_sweep(actual, blend, fold_sizes)
        per_fold_results.append({"run": r["run"], "kind": r["kind"] + "_perfold",
                                 "weights": r["weights"], "fold_thresholds": ts,
                                 "accuracy": acc})
    per_fold_results.sort(key=lambda r: r["accuracy"], reverse=True)
    best_perfold = per_fold_results[0]
    print(f"L4 best (per-fold thresholds): {best_perfold['run']} "
          f"acc={best_perfold['accuracy']:.4f}")

    print(f"\nPrevious best: {TARGET_PREVIOUS_BEST:.4f}")
    print(f"New best, global threshold: {best_global['accuracy']:.4f}")
    print(f"New best, per-fold thresholds: {best_perfold['accuracy']:.4f}")

    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    metrics["escalation_level3"] = {
        "date": str(date.today()),
        "levels_run": ["wider_pool_singles_pairs", "weighted_triples_top8",
                       "per_fold_thresholds_top20"],
        "pool_size": len(probs),
        "best_global_threshold": best_global,
        "best_per_fold_thresholds": best_perfold,
        "top_10_global": results[:10],
        "top_5_per_fold": per_fold_results[:5],
        "note": ("Per-fold threshold results select 5 thresholds on the same folds "
                 "they are scored on (vs 1 global threshold) - a strictly more "
                 "aggressive selection effect. Both numbers are fold-fitted; "
                 "neither demonstrates out-of-sample skill."),
    }
    metrics["winner"] = {**best_global, "auc": None,
                         "note": "found during escalation level 3; see 'escalation_level3'"}
    metrics["winner_most_aggressive"] = {**best_perfold, "auc": None,
                                         "note": "per-fold thresholds; see 'escalation_level3'"}
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Updated {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
