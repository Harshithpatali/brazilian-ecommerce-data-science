from __future__ import annotations

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


def eta_squared(groups: list[np.ndarray]) -> float:
    values = np.concatenate(groups)
    grand_mean = values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((values - grand_mean) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def one_way_anova(df: pd.DataFrame) -> None:
    work = df.dropna(subset=["segment", "monetary"]).copy()
    work["log_monetary"] = np.log1p(work["monetary"].clip(lower=0))
    groups = [g["log_monetary"].to_numpy() for _, g in work.groupby("segment")]

    statistic, p_value = stats.f_oneway(*groups)
    levene_stat, levene_p = stats.levene(*groups, center="median")
    effect = eta_squared(groups)

    print("\n=== One-way ANOVA: customer segment and monetary value ===")
    print(f"F-statistic: {statistic:.6f}")
    print(f"p-value:     {p_value:.6g}")
    print(f"eta-squared: {effect:.6f}")
    print(f"Levene W:    {levene_stat:.6f}")
    print(f"Levene p:    {levene_p:.6g}")
    print("Conclusion:", "reject H0" if p_value < ALPHA else "fail to reject H0")

    tukey = pairwise_tukeyhsd(
        endog=work["log_monetary"],
        groups=work["segment"],
        alpha=ALPHA,
    )
    print("\nTukey HSD post-hoc comparisons:")
    print(tukey)

    if levene_p < ALPHA:
        kw_stat, kw_p = stats.kruskal(*groups)
        print("\nVariance heterogeneity detected; Kruskal-Wallis sensitivity check:")
        print(f"H-statistic: {kw_stat:.6f}")
        print(f"p-value:     {kw_p:.6g}")


def two_factor_anova(df: pd.DataFrame) -> None:
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
    print("\n=== Two-factor ANOVA: segment x delivery performance -> review score ===")
    print(table)
    print("\nInteraction interpretation:")
    interaction_p = table.loc[
        "C(segment):C(delivery_performance_group)", "PR(>F)"
    ]
    if interaction_p < ALPHA:
        print("Evidence of an interaction effect at alpha=0.05.")
    else:
        print("No statistically significant interaction detected at alpha=0.05.")


def mean_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    mean = float(x.mean())
    se = float(stats.sem(x))
    margin = float(stats.t.ppf((1 + confidence) / 2, n - 1) * se) if n > 1 else np.nan
    return mean, mean - margin, mean + margin


def sampling_analysis(df: pd.DataFrame) -> None:
    population = df.dropna(subset=["segment", "monetary"]).copy()
    population["monetary"] = population["monetary"].clip(lower=0)

    population_mean, pop_low, pop_high = mean_ci(population["monetary"])
    n = min(1000, len(population))
    simple = population.sample(n=n, random_state=RANDOM_SEED)
    simple_mean, simple_low, simple_high = mean_ci(simple["monetary"])

    print("\n=== Sampling analysis: simple random sample ===")
    print(f"Population size: {len(population):,}")
    print(f"Sample size:     {n:,}")
    print(f"Population mean: {population_mean:.4f}")
    print(f"Sample mean:     {simple_mean:.4f}")
    print(f"Sample 95% CI:   [{simple_low:.4f}, {simple_high:.4f}]")
    print(f"Absolute error:  {abs(simple_mean - population_mean):.4f}")
    print(f"Relative error:  {abs(simple_mean - population_mean) / abs(population_mean):.4%}")

    # Proportionally allocated stratified sample across RFM segments.
    allocation = (
        population["segment"]
        .value_counts(normalize=True)
        .mul(n)
        .round()
        .astype(int)
    )
    # Adjust rounding so the requested sample size is respected.
    while allocation.sum() < n:
        allocation.iloc[np.argmax(population["segment"].value_counts(normalize=True).to_numpy())] += 1
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

    # Post-stratification weights = population share / sample share.
    pop_share = population["segment"].value_counts(normalize=True)
    sample_share = stratified["segment"].value_counts(normalize=True)
    weights = pop_share / sample_share
    weighted_mean = float(
        np.average(
            stratified["monetary"],
            weights=stratified["segment"].map(weights),
        )
    )

    print("\n=== Stratified sampling by RFM segment ===")
    print(f"Stratified sample size: {len(stratified):,}")
    print(f"Stratified sample mean: {stratified['monetary'].mean():.4f}")
    print(f"Weighted sample mean:   {weighted_mean:.4f}")
    print(f"Population mean:         {population_mean:.4f}")
    print(f"Weighted absolute error:{abs(weighted_mean - population_mean):.4f}")
    print("\nSegment allocation:")
    print(allocation.sort_index())


def main() -> None:
    print("Loading statistical-analysis datasets from PostgreSQL...")
    anova_df = query_to_dataframe(
        "SELECT * FROM analytics.customer_anova_dataset"
    )
    sampling_df = query_to_dataframe(
        "SELECT * FROM analytics.sampling_population"
    )

    one_way_anova(anova_df)
    two_factor_anova(anova_df)
    sampling_analysis(sampling_df)


if __name__ == "__main__":
    main()
