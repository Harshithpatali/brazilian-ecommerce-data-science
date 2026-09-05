from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from src.database.io import query_to_dataframe


RANDOM_SEED = 42
ALPHA = 0.05
ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_OUT = ROOT / "data" / "dashboard" / "statistical_results.json"


def eta_squared(groups: list[np.ndarray]) -> float:
    values = np.concatenate(groups)
    grand_mean = values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((values - grand_mean) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def one_way_anova(df: pd.DataFrame) -> dict:
    work = df.dropna(subset=["segment", "monetary"]).copy()
    work["log_monetary"] = np.log1p(work["monetary"].clip(lower=0))
    grouped = work.groupby("segment", observed=True)
    groups = [g["log_monetary"].to_numpy() for _, g in grouped]

    statistic, p_value = stats.f_oneway(*groups)
    levene_stat, levene_p = stats.levene(*groups, center="median")
    effect = eta_squared(groups)

    tukey = pairwise_tukeyhsd(
        endog=work["log_monetary"],
        groups=work["segment"],
        alpha=ALPHA,
    )
    tukey_table = pd.DataFrame(
        tukey._results_table.data[1:],
        columns=tukey._results_table.data[0],
    )
    for column in ["meandiff", "p-adj", "lower", "upper"]:
        tukey_table[column] = pd.to_numeric(tukey_table[column], errors="coerce")
    tukey_table["reject"] = tukey_table["reject"].astype(bool)

    result = {
        "f_statistic": float(statistic),
        "p_value": float(p_value),
        "eta_squared": float(effect),
        "levene_statistic": float(levene_stat),
        "levene_p_value": float(levene_p),
        "null_conclusion": "Reject H0" if p_value < ALPHA else "Fail to reject H0",
        "variance_heterogeneity": bool(levene_p < ALPHA),
        "tukey": tukey_table.to_dict(orient="records"),
    }

    if levene_p < ALPHA:
        kw_stat, kw_p = stats.kruskal(*groups)
        result["kruskal_wallis"] = {
            "h_statistic": float(kw_stat),
            "p_value": float(kw_p),
        }

    print("\n=== One-way ANOVA: customer segment and monetary value ===")
    print(f"F-statistic: {statistic:.6f}")
    print(f"p-value:     {p_value:.6g}")
    print(f"eta-squared: {effect:.6f}")
    print(f"Levene W:    {levene_stat:.6f}")
    print(f"Levene p:    {levene_p:.6g}")
    print("Conclusion:", result["null_conclusion"])
    print("\nTukey HSD post-hoc comparisons:")
    print(tukey)

    if levene_p < ALPHA:
        print("\nVariance heterogeneity detected; Kruskal-Wallis sensitivity check:")
        print(f"H-statistic: {kw_stat:.6f}")
        print(f"p-value:     {kw_p:.6g}")

    return result


def two_factor_anova(df: pd.DataFrame) -> dict:
    work = df.dropna(
        subset=["segment", "delivery_performance_group", "avg_review_score"]
    ).copy()
    work["segment"] = work["segment"].astype("category")
    work["delivery_performance_group"] = work["delivery_performance_group"].astype(
        "category"
    )

    model = ols(
        "avg_review_score ~ C(segment) * C(delivery_performance_group)",
        data=work,
    ).fit()
    table = anova_lm(model, typ=2)
    interaction_term = "C(segment):C(delivery_performance_group)"
    interaction_p = float(table.loc[interaction_term, "PR(>F)"])

    result = {
        "anova_table": table.reset_index(names="term").to_dict(orient="records"),
        "interaction_p_value": interaction_p,
        "interaction_significant": bool(interaction_p < ALPHA),
        "interaction_interpretation": (
            "Evidence of an interaction effect at alpha=0.05."
            if interaction_p < ALPHA
            else "No statistically significant interaction detected at alpha=0.05."
        ),
    }

    print("\n=== Two-factor ANOVA: segment x delivery performance -> review score ===")
    print(table)
    print("\nInteraction interpretation:")
    print(result["interaction_interpretation"])

    return result


def mean_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    mean = float(x.mean())
    se = float(stats.sem(x))
    margin = float(stats.t.ppf((1 + confidence) / 2, n - 1) * se) if n > 1 else np.nan
    return mean, mean - margin, mean + margin


def sampling_analysis(df: pd.DataFrame) -> dict:
    population = df.dropna(subset=["segment", "monetary"]).copy()
    population["monetary"] = population["monetary"].clip(lower=0)

    population_mean, _, _ = mean_ci(population["monetary"])
    n = min(1000, len(population))
    simple = population.sample(n=n, random_state=RANDOM_SEED)
    simple_mean, simple_low, simple_high = mean_ci(simple["monetary"])
    simple_abs_error = abs(simple_mean - population_mean)
    simple_relative_error = simple_abs_error / abs(population_mean)

    allocation = (
        population["segment"]
        .value_counts(normalize=True)
        .mul(n)
        .round()
        .astype(int)
    )
    proportions = population["segment"].value_counts(normalize=True)
    while allocation.sum() < n:
        allocation.loc[proportions.idxmax()] += 1
    while allocation.sum() > n:
        eligible = allocation[allocation > 1]
        allocation.loc[eligible.index[0]] -= 1

    stratified_parts = []
    for segment, target in allocation.items():
        frame = population[population["segment"] == segment]
        stratified_parts.append(
            frame.sample(n=min(int(target), len(frame)), random_state=RANDOM_SEED)
        )
    stratified = pd.concat(stratified_parts, ignore_index=True)

    pop_share = population["segment"].value_counts(normalize=True)
    sample_share = stratified["segment"].value_counts(normalize=True)
    weights = pop_share / sample_share
    weighted_mean = float(
        np.average(
            stratified["monetary"],
            weights=stratified["segment"].map(weights),
        )
    )

    result = {
        "population_size": int(len(population)),
        "sample_size": int(n),
        "population_mean": population_mean,
        "simple_random_sample": {
            "sample_mean": simple_mean,
            "ci_lower": simple_low,
            "ci_upper": simple_high,
            "absolute_error": float(simple_abs_error),
            "relative_error": float(simple_relative_error),
        },
        "stratified_sample": {
            "sample_size": int(len(stratified)),
            "sample_mean": float(stratified["monetary"].mean()),
            "weighted_mean": weighted_mean,
            "weighted_absolute_error": float(abs(weighted_mean - population_mean)),
            "allocation": {str(k): int(v) for k, v in allocation.sort_index().items()},
        },
    }

    print("\n=== Sampling analysis: simple random sample ===")
    print(f"Population size: {len(population):,}")
    print(f"Sample size:     {n:,}")
    print(f"Population mean: {population_mean:.4f}")
    print(f"Sample mean:     {simple_mean:.4f}")
    print(f"Sample 95% CI:   [{simple_low:.4f}, {simple_high:.4f}]")
    print(f"Absolute error:  {simple_abs_error:.4f}")
    print(f"Relative error:  {simple_relative_error:.4%}")

    print("\n=== Stratified sampling by RFM segment ===")
    print(f"Stratified sample size: {len(stratified):,}")
    print(f"Stratified sample mean: {stratified['monetary'].mean():.4f}")
    print(f"Weighted sample mean:   {weighted_mean:.4f}")
    print(f"Population mean:         {population_mean:.4f}")
    print(f"Weighted absolute error:{abs(weighted_mean - population_mean):.4f}")
    print("\nSegment allocation:")
    print(allocation.sort_index())

    return result


def run_analysis() -> dict:
    print("Loading statistical-analysis datasets from PostgreSQL...")
    anova_df = query_to_dataframe("SELECT * FROM analytics.customer_anova_dataset")
    sampling_df = query_to_dataframe("SELECT * FROM analytics.sampling_population")

    results = {
        "alpha": ALPHA,
        "random_seed": RANDOM_SEED,
        "one_way_anova": one_way_anova(anova_df),
        "two_factor_anova": two_factor_anova(anova_df),
        "sampling": sampling_analysis(sampling_df),
    }
    return results


def _json_safe(value):
    """Convert NumPy/Pandas scalars and non-finite floats to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_dashboard_results(results: dict) -> Path:
    DASHBOARD_OUT.parent.mkdir(parents=True, exist_ok=True)
    safe_results = _json_safe(results)
    with DASHBOARD_OUT.open("w", encoding="utf-8") as handle:
        json.dump(safe_results, handle, indent=2, allow_nan=False)
    print(f"Wrote {DASHBOARD_OUT}")
    return DASHBOARD_OUT


def main() -> None:
    results = run_analysis()
    save_dashboard_results(results)


if __name__ == "__main__":
    main()
