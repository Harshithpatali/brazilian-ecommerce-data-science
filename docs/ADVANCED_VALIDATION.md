# End-to-end validation: advanced customer science

The first-order 180-day repeat-purchase view is included in the normal SQL build (python -m scripts.build_all).

After creating the view, the build runs a read-only database validation with python -m scripts.validate_advanced_customer_science.

The validator checks that the view exists, has one row per stable customer, contains no null customer/order keys or invalid labels, excludes customers without a fully observable 180-day outcome window, and has internally consistent repeat labels and time-to-second-purchase values. It prints the mature cohort size, observed 180-day repeat rate and index-date range.

These checks establish target-data integrity only. They do not establish model quality, calibrated CLV, or causal effects. BG/NBD and Gamma-Gamma still require a separately defined calibration/holdout evaluation using delivered-order transactions and a later observation window.

The connected Olist database is not available to GitHub Actions unless database credentials and an explicit integration-test environment are configured, so the live database check must be run in the configured project environment.
