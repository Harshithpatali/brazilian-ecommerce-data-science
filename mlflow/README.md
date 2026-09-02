# MLflow

This project uses MLflow for local experiment tracking.

Default tracking URI:

```text
file:./mlruns
```

Run the ML training pipeline:

```bash
python -m scripts.run_ml
```

The pipeline logs:

- experiment name: `brazilian-ecommerce-churn`
- model family
- hyperparameters
- cross-validation metrics
- holdout metrics
- feature list
- selected model
- model artifact
- model comparison table

The local MLflow tracking directory is intentionally excluded from Git because it can become large.

To launch the local MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open the local MLflow URL shown by the terminal.

For a shared/server deployment later, set:

```env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=brazilian-ecommerce-churn
```
