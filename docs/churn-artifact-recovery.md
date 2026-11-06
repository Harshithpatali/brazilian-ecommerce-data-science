# Churn artifact contract recovery

The Churn Intelligence page validates the complete model contract before loading the
estimator. A valid metadata file alone is not sufficient: the model binary, feature
schema, training dataset, and evaluation outputs must come from the same successful
training run.

## Recovery

1. Configure the PostgreSQL connection secrets used by the workflow.
2. Run the database-backed `Rebuild and retrain customer churn` workflow on a
   single intended branch. It refreshes the SQL feature table, validates its
   snapshot grain and label windows, exports the parquet dataset, trains/calibrates,
   validates metadata and binary fingerprint, and commits the generated artifacts.
3. Confirm the workflow's final commit contains both `models/model_metadata.json`
   and `models/best_churn_model.joblib`, with the model SHA-256 matching metadata.
4. Redeploy/restart the Streamlit app so the filesystem and Streamlit resource
   cache load the same published artifact set.

Do not edit `model_version`, `feature_columns`, target-class text, threshold
policy, or the SHA-256 value by hand to silence a warning. Those fields describe
the actual estimator and evaluation run. A metadata-only edit can make the page
look ready while leaving the model binary or dataset incompatible.

## Important

The current Streamlit scoring path expects the positive repeat-purchase probability
in probability column 0 for the corrected repeat-purchase wrapper. The legacy
classification path used churn probability in column 1 and inverted it. Keep the
metadata class definition and the trained estimator's `classes_` / probability
column ordering aligned. Re-evaluate the final holdout after any target or feature
contract change; do not reuse metrics from a previous training run.
