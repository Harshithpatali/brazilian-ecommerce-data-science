# System architecture

```mermaid
flowchart TD
    A[Olist source tables] --> B[PostgreSQL source validation]
    B --> C[SQL data model and KPI marts]
    C --> D[Customer, cohort, product, seller and delivery datasets]
    C --> E[Point-in-time monthly customer snapshots]
    D --> F[CSV dashboard artifacts]
    E --> G[Temporal ML training and evaluation]
    G --> H[Model metadata, comparisons and holdout diagnostics]
    F --> I[Streamlit analytics pages]
    H --> I
    D --> J[RFM quintiles and customer explanations]
    D --> K[Kaplan-Meier time-to-second-purchase]
    J --> L[Campaign scenario planner]
    K --> I
    L --> I
    I --> M[Human decision support]
```

## Trust boundaries and interpretation
- Source timestamps and delivered-order definitions are validated before SQL rebuilds.
- ML predictors are point-in-time; future orders are used only to define the outcome.
- Temporal partitions use a 90-day label-horizon embargo and the final test is held out from model selection.
- RFM segments are descriptive rules and relative scores, not causal personas.
- Survival analysis treats customers without a second observed purchase by the data end as right-censored.
- Campaign lift and unit economics are explicit scenario assumptions. No randomized campaign exposure exists in Olist, so outputs are not causal uplift estimates.
- v6 remains research-only because holdout PR-AUC (0.01048) did not beat lifetime-frequency (0.01665).
