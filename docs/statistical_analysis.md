# Statistical Analysis Plan

The SQL layer prepares analysis-ready data; Python performs formal statistical tests.

## Research questions

1. Is delivery delay associated with lower review scores?
2. Do customer segments differ materially in purchase frequency and monetary value?
3. Does customer segment or delivery performance relate to review score, and is there an interaction between the two factors?
4. How accurately can customer-level population metrics be estimated from simple random and stratified samples?
5. How does post-stratification weighting affect sample-based estimates when segment proportions differ from the population?
6. Do cohorts differ in retention?

## ANOVA analysis

### One-way ANOVA

**Null hypothesis:** mean customer monetary value is equal across RFM segments.

**Alternative hypothesis:** at least one segment has a different mean customer monetary value.

The implementation uses `log1p(monetary)` for the ANOVA scale because monetary values are strongly right-skewed in typical e-commerce data. The analysis reports the F-statistic, p-value, eta-squared effect size, Levene's variance test, and Tukey HSD post-hoc comparisons. A Kruskal-Wallis sensitivity check is reported when the homogeneity-of-variance assumption is not supported.

### Two-factor ANOVA

**Outcome:** average review score at customer level.

**Factors:** RFM customer segment and delivery-performance group.

The model evaluates both main effects and the segment × delivery-performance interaction. The interaction is the key test for whether the association between customer segment and review score differs across delivery-performance groups.

## Sampling analysis

The sampling population is one row per customer from `analytics.customer_rfm` with non-null frequency, monetary value, and segment.

The Python analysis compares:

- a deterministic simple random sample of up to 1,000 customers;
- a proportionally allocated stratified sample by RFM segment;
- population and sample means;
- 95% confidence intervals for sample means;
- absolute and relative estimation error;
- post-stratification weighted estimates using population-share / sample-share weights.

The sampling analysis is descriptive and does not claim to reproduce Nielsen's production survey-sampling, weighting, or variance-estimation methodology.

## Statistical quality principles

- Define hypotheses before inspecting results.
- Choose tests based on data-generating assumptions.
- Check assumptions and report sensitivity analyses where appropriate.
- Report effect size and uncertainty, not only p-values.
- Avoid causal language unless the study design supports causality.
- Treat statistical significance and practical significance separately.
- Use multiple-comparison correction where a family of related tests is evaluated.
- Use a fixed random seed for reproducible sampling examples.

## Running the analysis

First build the SQL analytical layer so that `analytics.customer_anova_dataset` and `analytics.sampling_population` exist:

```bash
python -m scripts.build_all
```

Then run:

```bash
python -m src.statistical_analysis
```

The script reads the analysis-ready views through the project's existing SQLAlchemy/PostgreSQL connection layer and prints the statistical results. It does not modify raw source tables.
