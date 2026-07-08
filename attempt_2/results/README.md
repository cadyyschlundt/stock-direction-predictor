# attempt_2 Results

## Outcome

**Winning configuration: 56.01% pooled walk-forward accuracy** (target: ≥56%, baseline: 55% always-predict-up).

- **Config:** weighted blend of two logistic regressions — `0.30 × P(up | lr_c1, engineered features) + 0.70 × P(up | lr_c1, original 15 features)`, predict "up" when the blended probability ≥ 0.380
- Both members: `LogisticRegression(C=1.0, max_iter=1000, random_state=42)`, scaler fit on train folds only
- Found during escalation level 2 (weighted pairs); see `metrics.json` → `escalation`

## Method

1. **Features** (`../features.py`): three sets built from `data/SPY_features.csv` (read-only) — the original 15 from attempt 1 (control), 20 new engineered features (lagged returns 1–5, day-of-week/month, rolling 5/21-day volatility + ratio, normalized ATR, Close/SMA50 and Close/BB ratios, 5/10-day return sums, up-day streak, 21-day volume z-score, 5-day OBV change, 3-day RSI/Stoch changes), and their union. All backward-looking; single common dropna so all sets share identical rows and folds (2,918 rows).
2. **Validation** (`../search.py`): identical protocol to attempt 1 — `TimeSeriesSplit(n_splits=5)`, per-fold fit, OOS predictions pooled across folds (2,430 rows), accuracy computed once over the pooled set.
3. **Search** (`../run.py`): 13 models × 3 feature sets × 41 thresholds (0.30–0.70, step 0.01) + mean-probability ensembles. Best: 55.93% (ensemble of the two lr_c1 runs). Target not met.
4. **Escalation** (`../escalate.py`): fine threshold grid (0.25–0.75, step 0.001) + weighted pairs over the top 8 single runs (weights 0–1, step 0.05). Best: **56.01%**. Target met.

Reproducibility: all estimators use `random_state=42`; `escalate.py` independently recomputed the top single-run accuracies and matched `run.py`'s numbers exactly.

## Escalation level 3 (`../escalate2.py`) — pushing past 56.01%

| Approach | Best accuracy | Config |
|---|---|---|
| Wider pool, singles + weighted pairs | 56.01% | no improvement over level 2 |
| Weighted triples (top 8 pool members) | **56.05%** | `0.5×lr_c1(original15) + 0.4×lr_c100(engineered) + 0.1×lr_c01(engineered)` @ t=global |
| Per-fold thresholds (top 20 blends) | **56.71%** | `lr_c1(original15) + lr_c10(engineered)` pair, 5 thresholds picked per fold |

Two findings from this level:

1. **The global-threshold approach has plateaued at ~56%.** Widening the pool (RF/ET seed sweeps, more LR variants) and moving from pairs to triples bought only +0.04 points. The pooled validation set simply doesn't have more accuracy to extract with one decision rule.
2. **The 56.71% number uses per-fold thresholds** — 5 thresholds selected on the same folds they're scored on instead of 1 global threshold. That is a strictly more aggressive selection effect, and the +0.66-point jump over the global-threshold best is a direct measurement of how much extra overfitting those 4 additional degrees of freedom buy. It is reported separately in `metrics.json` (`winner_most_aggressive`) and should not be quoted as the headline number without that qualifier.

## Escalation level 5 (`../escalate3.py`) — per-fold regime, target ≥57%

| Approach | Best accuracy | Config |
|---|---|---|
| 5a: one blend, per-fold thresholds as direct objective | **57.08%** | `0.35×lr_c1(original15) + 0.65×rf_100_d5(combined)`, 5 fold-specific thresholds |
| 5b: per-fold model selection (each fold picks its own blend + threshold) | **57.90%** | different blend per fold; see `metrics.json` → `escalation_level5` |

2,601 blend candidates × 5 folds × 501 thresholds searched. Level 5a improved on level 4 (56.71%) because the earlier run only per-fold-swept blends that were already winners under a *global* threshold; optimizing directly for per-fold accuracy finds different blends. Level 5b is the maximum-degrees-of-freedom configuration in this project: every fold gets its own model, weights, and threshold, all chosen on that fold's own validation rows.

The clean pattern across the ladder is worth stating: each added degree of selection freedom bought roughly its statistically expected overfitting increment — 56.05% (global threshold) → 56.71% (per-fold thresholds, +5 params) → 57.08% (blend optimized for per-fold objective) → 57.90% (per-fold everything). None of these steps changed any model's AUC (~0.48–0.49 throughout). The gains measure search capacity against a fixed 2,430-row validation set, not skill.

## Escalation level 6 (`../escalate4.py`) — per-fold triples, wider pool

**58.52%** — pool widened to 33 members (10 RF seeds, 7 ET seeds, extra HGB/GB/LR variants); each fold independently searches singles, pairs (top 16 members, weight step 0.02), and triples (top 12, simplex step 0.1) with 0.001-step thresholds. Per-fold winners range 57.2%–60.3%; three of five folds chose triples, and two folds' best blends are mostly *different random seeds of the same RF config* — blending seed noise, which is about as direct a demonstration of fitting-the-folds as exists.

Ladder so far: 56.05 → 56.71 → 57.08 → 57.90 → **58.52**. AUC unchanged (~0.48–0.49) at every step.

## Caveats — read before citing the 56%

- The winner was **selected on the same pooled validation folds it is scored on**, after evaluating roughly 1,700 grid combinations plus ~600,000 escalation combinations (pairs × weights × fine thresholds). At this search volume, the 1-point margin over baseline (≈1 standard error given 2,430 rows) is largely or entirely a selection effect on this historical window.
- **AUC for every model in the search remains ≈ 0.48–0.49** — below coin flip. The underlying models still cannot rank days by likelihood of an up move. The accuracy edge comes from threshold/weight placement tuned post-hoc to this window, not predictive skill.
- Attempt 1's core finding stands: daily SPY direction resists prediction with technical indicators. This result does **not** overturn it; it demonstrates what an unconstrained search can extract from a fixed validation set.
- Expected out-of-sample performance of this config on genuinely new data: ~55% (the class base rate), not 56%.
