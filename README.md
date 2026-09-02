# Brazilian E-Commerce Data Science Intelligence Platform

An end-to-end portfolio project built on the Brazilian Olist e-commerce dataset stored in Supabase PostgreSQL.

## Architecture

```text
Supabase PostgreSQL
        |
        v
   SQL / Analytics
        |
        +--------------------+
        |                    |
        v                    v
Dashboard datasets      ML feature dataset
        |                    |
        v                    v
   Streamlit             Python / ML
        \                    /
         \                  /
          +------ GitHub --+
```

## Project goals

1. Establish a trustworthy understanding of the relational data.
2. Profile data quality and validate logical relationships.
3. Build reusable analytical views/tables without changing raw `public.olist_*` tables.
4. Answer business questions across revenue, customers, products, sellers, delivery, reviews and geography.
5. Build RFM and cohort-retention analysis.
6. Create a customer feature table for churn modeling.
7. Present decision-ready results through Streamlit.
8. Keep SQL and application logic version-controlled and reproducible.

## Source tables

- `public.olist_customers`
- `public.olist_orders`
- `public.olist_order_items`
- `public.olist_order_payments`
- `public.olist_order_reviews`
- `public.olist_products`
- `public.olist_sellers`
- `public.olist_geolocation`
- `public.product_category_name_translation`

## Important modeling rule

The project explicitly respects table grain:

- `olist_orders`: approximately one row per order
- `olist_order_items`: one row per order item
- `olist_order_payments`: one row per payment installment/sequence
- `olist_order_reviews`: review-level rows
- `olist_customers`: order-to-customer records
- `customer_unique_id`: the customer identity used for customer-level analysis

Do not join order items and payments directly and then sum both measures without first aggregating them to order grain.

## Setup

### 1. Create environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Supabase credentials

Copy `.env.example` to `.env`.

Use either:

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/postgres
```

or the individual variables:

```env
SUPABASE_DB_HOST=
SUPABASE_DB_PORT=5432
SUPABASE_DB_NAME=postgres
SUPABASE_DB_USER=
SUPABASE_DB_PASSWORD=
```

Never commit `.env`.

### 4. Validate the connection

```bash
python scripts/check_connection.py
```

### 5. Create project schemas/views

Run:

```bash
python scripts/run_sql_folder.py sql/03_data_model
python scripts/run_sql_folder.py sql/04_kpi_layer
python scripts/run_sql_folder.py sql/05_customer_analytics
python scripts/run_sql_folder.py sql/06_cohort_retention
python scripts/run_sql_folder.py sql/07_product_analytics
python scripts/run_sql_folder.py sql/08_seller_analytics
python scripts/run_sql_folder.py sql/09_delivery_analytics
python scripts/run_sql_folder.py sql/10_customer_experience
python scripts/run_sql_folder.py sql/11_geographic_analytics
python scripts/run_sql_folder.py sql/12_advanced_analytics
python scripts/run_sql_folder.py sql/14_ml_features
python scripts/run_sql_folder.py sql/15_reporting
```

The scripts are ordered and designed to be rerunnable.

### 6. Build local dashboard datasets

```bash
python scripts/build_dashboard_data.py
```

This writes reusable datasets to `data/dashboard/`.

### 7. Build ML dataset

```bash
python scripts/build_ml_data.py
```

This writes `data/ml/customer_features.parquet`.

### 8. Run Streamlit

```bash
streamlit run app/streamlit_app.py
```

## Suggested workflow

1. Run the investigation SQL.
2. Run the data-quality SQL.
3. Read `docs/data_model.md`.
4. Build/validate the analytics schema.
5. Generate dashboard datasets.
6. Review the Streamlit app.
7. Run statistical analysis in Python.
8. Train the churn model.
9. Document business recommendations.
10. Commit SQL, Python and documentation to GitHub.

## Security

`.env`, local database credentials and generated model artifacts are ignored by git. Use `.env.example` as the safe template.

## Portfolio positioning

This project is designed to demonstrate:

- Advanced PostgreSQL/SQL
- Data-quality analysis
- Analytical data modeling
- Window functions and CTEs
- Customer segmentation and cohort analysis
- Statistical reasoning
- Feature engineering
- Machine learning
- Business communication
- Reproducible engineering practices


## Full ML workflow

After the SQL/analytics build:

```bash
python -m scripts.run_ml
```

This:

1. Builds `data/ml/customer_churn_dataset.parquet`
2. Trains the chronological Logistic Regression baseline
3. Writes model artifacts to `models/`

Then launch Streamlit:

```bash
python -m streamlit run app/streamlit_app.py
```

The ML dashboard is available at:

```text
/churn
```

The churn model is trained with a chronological holdout and predicts the probability that a customer makes no delivered purchase during the next 90 days.


## MLflow experiment tracking

The tuned churn pipeline tracks all two candidate model families and their hyperparameter searches in MLflow.

Run:

```bash
python -m scripts.run_ml
```

Launch the local tracking UI:

```bash
python -m mlflow ui --backend-store-uri ./mlruns
```

The selected fitted model is also saved locally as:

```text
models/best_churn_model.joblib
```

No FastAPI service is required; Streamlit loads the saved fitted pipeline directly.


Windows helper:

```bash
python -m scripts.start_mlflow
```

## Optimized churn build

The ML dataset no longer performs a snapshot × full customer-history join. It materializes delivered orders, aggregates to customer-month, then creates the point-in-time dataset from the compact table.

After replacing an older project version, run:

```bash
python -m scripts.build_all
python -m scripts.run_ml
python -m streamlit run app/streamlit_app.py
```


## Optimized churn workflow

The churn dataset is materialized in PostgreSQL using a compact customer-month table. This avoids repeated raw-order snapshot joins and reduces the chance of Supabase statement timeouts.

Recommended run order:

```bash
python -m scripts.build_all
python -m scripts.validate_churn_data
python -m scripts.run_ml
python -m scripts.start_mlflow
python -m streamlit run app/streamlit_app.py
```
