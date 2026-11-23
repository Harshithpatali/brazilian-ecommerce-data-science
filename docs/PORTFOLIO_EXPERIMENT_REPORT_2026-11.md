# Repeat-purchase model experiment report

**Report status:** reproducible evaluation record; no new training run is claimed by this document.  
**Target:** second delivered purchase within 90 days of a customer-month snapshot.  
**Evaluation policy:** chronological splits with a 90-day label-window purge; final holdout is frozen.

## Current evidence (v6)

The existing investigation artifact records the following final untouched-holdout results:

| Metric | Random Forest | Lifetime-frequency baseline |
|---|---:|---:|
| PR-AUC | 0.01048 | 0.01665 |
| Precision at 5% contact capacity | 1.21% | 1.69% |
| Precision at 10% contact capacity | 1.01% | 1.21% |
| Precision at 20% contact capacity | 0.87% | 0.88% |

The selected model's temporal-CV PR-AUC was 0.01196. This did not translate into a holdout win. The artifact remains **research-only**. Do not describe it as a promoted propensity model or tune against this holdout.

Source of truth: [Model Investigation](MODEL_INVESTIGATION.md) and [Model Card](MODEL_CARD.md).

## Next experiment: candidate development without holdout reuse

1. Rebuild the SQL feature table from source using `python -m scripts.build_all` after configuring the documented database environment.
2. Run `python -m pytest -q` before training.
3. Run `python -m src.models.train_churn` (or the repository's documented training entry point) and retain the generated MLflow run, split metadata, model artifact, and baseline comparison.
4. Treat the existing final test window as a historical diagnostic only. Candidate development, feature selection, calibration and threshold selection must use training/validation periods.
5. If a candidate is selected, evaluate it on a **new, later, fully matured temporal window** with a new 90-day embargo. Do not relabel the old test set as validation.
6. Compare against the same baselines at PR-AUC and 5%, 10%, and 20% monthly contact capacity. Report prevalence and confidence intervals alongside point estimates.

## Promotion gate

A candidate can be promoted only if the new untouched temporal holdout beats the lifetime-frequency baseline on PR-AUC and precision at each predeclared contact capacity, passes leakage and calibration checks, and has stable performance across date-grouped folds. A failure is a valid research result.

## Reproducibility record

Record the following with every run:

- Git commit SHA and Python/dependency versions.
- Source data snapshot/cutoff and feature-table fingerprint.
- Target definition, horizon and 90-day purge dates.
- Fold boundaries, event prevalence and per-fold metrics.
- Candidate model/configuration, random seeds and MLflow run ID.
- Baseline results at identical evaluation rows.
- Holdout decision and whether the artifact is research-only or eligible.

**Interpretation:** the present result demonstrates temporal evaluation discipline and a meaningful baseline failure diagnosis. It does not demonstrate that the current machine-learning model improves campaign targeting.
