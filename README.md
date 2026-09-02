<div align="center">

# 🇧🇷 Brazilian E-Commerce Intelligence Platform

**An end-to-end, SQL-first data science platform** built on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — data modeling, business analytics, statistical analysis, machine learning (customer churn), and a live interactive dashboard.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white&style=for-the-badge)](https://olist-ecommerce-ds.streamlit.app/)

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supabase-336791?logo=postgresql&logoColor=white)](https://supabase.com/)
[![scikit--learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)

[![Last Commit](https://img.shields.io/github/last-commit/Harshithpatali/brazilian-ecommerce-data-science?color=blue)](https://github.com/Harshithpatali/brazilian-ecommerce-data-science/commits)
[![Repo Size](https://img.shields.io/github/repo-size/Harshithpatali/brazilian-ecommerce-data-science?color=informational)](https://github.com/Harshithpatali/brazilian-ecommerce-data-science)
[![Stars](https://img.shields.io/github/stars/Harshithpatali/brazilian-ecommerce-data-science?style=social)](https://github.com/Harshithpatali/brazilian-ecommerce-data-science/stargazers)

</div>

---

## 📑 Table of Contents

- [Dashboard Preview](#-dashboard-preview)
- [Overview](#-overview)
- [Key Highlights](#-key-highlights)
- [Architecture](#️-architecture)
- [Live Dashboard](#️-live-dashboard)
- [Source Data](#️-source-data)
- [Data Model](#-data-model)
- [SQL Layer](#-sql-layer)
- [Machine Learning — Customer Churn](#-machine-learning--customer-churn)
- [Statistical Analysis](#-statistical-analysis)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Testing](#-testing)
- [Security](#-security)
- [What This Project Demonstrates](#-what-this-project-demonstrates)
- [Documentation](#-documentation)
- [License](#-license)
- [Author](#-author)

---

## 🖼 Dashboard Preview

<table>
<tr>
<td width="50%">

**Executive Overview**
![Executive Overview](screenshots/UI.jpeg)
Top-line revenue, order volume, AOV, and marketplace KPIs at a glance.

</td>
<td width="50%">

**Customer Intelligence**
![Customer Intelligence](screenshots/UI1.jpeg)
RFM segmentation and customer value distribution.

</td>
</tr>
<tr>
<td width="50%">

**Retention & Cohorts**
![Retention](screenshots/UI2.jpeg)
Monthly cohort retention curves and behavior over time.

</td>
<td width="50%">

**Product Analytics**
![Products](screenshots/UI3.jpeg)
Category-level revenue, volume, and review quality.

</td>
</tr>
</table>

<div align="center">

**👉 Explore it yourself: [olist-ecommerce-ds.streamlit.app](https://olist-ecommerce-ds.streamlit.app/)**

</div>

---

## 📌 Overview

This project simulates the work of a senior data scientist embedded in an e-commerce marketplace. Starting from raw relational tables hosted in **Supabase PostgreSQL**, it builds a governed analytical data model in pure SQL, layers on business KPIs, customer segmentation, cohort retention, statistical hypothesis testing, and a **leakage-safe, point-in-time customer churn model**, and finally ships all of it as an interactive **Streamlit** application.

The guiding principle throughout the project is **SQL does the heavy lifting, Python does the modeling and presentation** — raw source tables are never mutated, every derived table respects the correct grain (order-level vs. item-level vs. payment-level), and the churn label is built with strict temporal leakage control.

## ✨ Key Highlights

<div align="center">

| 🗄️ 16 | 🧮 20+ | 🤖 3 | 📊 7 | 📈 90-day |
|:---:|:---:|:---:|:---:|:---:|
| SQL build stages | analytical tables/views | ML model families compared | live dashboard pages | churn prediction horizon |

</div>

- ✅ **Zero raw-table mutation** — every transformation lives in `staging` → `analytics` → `ml` schemas
- ✅ **Leakage-safe ML** — point-in-time features + chronological holdout, not random train/test splits
- ✅ **Experiment-tracked** — every model run logged and comparable in MLflow
- ✅ **Reproducible from scratch** — one command (`python -m scripts.build_all`) rebuilds the entire analytical layer
- ✅ **Shipped, not just notebooked** — deployed as a live, public Streamlit application

## 🏗️ Architecture

```mermaid
flowchart TD
    A[("🗄️ Supabase PostgreSQL<br/>raw olist_* tables")] --> B["🧱 SQL Analytical Layer<br/>staging → analytics → ml"]
    B --> C["📊 Dashboard-ready datasets"]
    B --> D["🎯 Point-in-time ML feature dataset"]
    C --> E["🖥️ Streamlit Dashboard"]
    D --> F["🤖 scikit-learn models<br/>+ MLflow tracking"]
    F --> G["💾 best_churn_model.joblib"]
    G --> E
    E --> H(("🌐 Live App<br/>olist-ecommerce-ds.streamlit.app"))

    style A fill:#336791,color:#fff
    style B fill:#4c72b0,color:#fff
    style E fill:#FF4B4B,color:#fff
    style F fill:#0194E2,color:#fff
    style H fill:#2ea44f,color:#fff
```

**Design goals:**

1. Establish a trustworthy, well-documented understanding of the raw relational data.
2. Profile data quality and validate logical/grain relationships before building on top of them.
3. Build reusable analytical views/tables **without ever mutating** the raw `public.olist_*` tables.
4. Answer real business questions across revenue, customers, products, sellers, delivery, reviews, and geography.
5. Build RFM segmentation and cohort-retention analysis.
6. Engineer a **leakage-safe, point-in-time** customer feature table for churn modeling.
7. Compare multiple model families with proper temporal validation and track experiments in MLflow.
8. Present everything through a decision-ready Streamlit dashboard.
9. Keep SQL, Python, and documentation fully version-controlled and reproducible from a clean database.

## 🖥️ Live Dashboard

👉 **https://olist-ecommerce-ds.streamlit.app/**

The dashboard includes the following pages:

| Page | What it shows |
|---|---|
| **Executive** | Top-line revenue, order volume, AOV, and marketplace KPIs over time |
| **Customers** | RFM segmentation, customer value distribution, repeat-purchase behavior |
| **Retention** | Monthly cohort retention curves |
| **Products** | Category-level performance — revenue, volume, and review quality |
| **Sellers** | Seller performance leaderboard and revenue concentration |
| **Delivery** | Delivery time distributions and delay analysis by region/seller |
| **Churn** | Interactive churn-risk scoring powered by the trained ML pipeline |

## 🗂️ Source Data

Raw tables live in the `public` schema of a Supabase Postgres instance:

- `public.olist_customers`
- `public.olist_orders`
- `public.olist_order_items`
- `public.olist_order_payments`
- `public.olist_order_reviews`
- `public.olist_products`
- `public.olist_sellers`
- `public.olist_geolocation`
- `public.product_category_name_translation`

### ⚠️ A critical modeling rule

This project explicitly respects table **grain**:

- `olist_orders` — approximately one row per order
- `olist_order_items` — one row per order **item**
- `olist_order_payments` — one row per payment installment/sequence
- `olist_order_reviews` — review-level rows
- `olist_customers` — order-to-customer records
- `customer_unique_id` — the **stable customer identity** used for all customer-level analysis (not `customer_id`, which is per-order)

**Order items and payments are never joined directly and summed** — both are first aggregated to order grain in `analytics.order_level` to avoid many-to-many row multiplication, a common and easy-to-miss bug in e-commerce data modeling.

## 🧱 Data Model

```mermaid
flowchart LR
    RAW[("public raw tables<br/>(untouched)")] --> OL["analytics.order_level<br/><i>central order-grain fact table</i>"]

    OL --> MR[monthly_revenue]
    OL --> CM[customer_metrics]
    OL --> RFM[customer_rfm]
    OL --> COH[cohort_retention]
    OL --> CAT[category_performance]
    OL --> SEL[seller_performance]
    OL --> DEL[delivery_performance]
    OL --> DVR[delivery_vs_review]
    OL --> GEO[geographic_performance]

    OL --> DO[ml.delivered_orders]
    DO --> CMA[ml.customer_monthly_activity]
    CMA --> CCD[("ml.customer_churn_dataset")]

    style RAW fill:#336791,color:#fff
    style OL fill:#4c72b0,color:#fff
    style CCD fill:#0194E2,color:#fff
```

See [`docs/data_model.md`](docs/data_model.md) for the full breakdown.

## 🔍 SQL Layer

All analytical logic is organized as **ordered, idempotent SQL scripts**, runnable end-to-end or individually:

```text
sql/
├── 01_database_investigation/   # schema/table discovery, row counts, relationships
├── 02_data_quality/             # null checks, duplicate checks, referential integrity
├── 03_data_model/                # schemas + analytics.order_level
├── 04_kpi_layer/                 # monthly revenue, executive KPIs, customer metrics
├── 05_customer_analytics/        # RFM scoring + customer segments
├── 06_cohort_retention/          # monthly cohort retention
├── 07_product_analytics/         # category performance
├── 08_seller_analytics/          # seller performance
├── 09_delivery_analytics/        # delivery time & delay performance
├── 10_customer_experience/       # delivery delay vs. review score
├── 11_geographic_analytics/      # state-level performance
├── 12_advanced_analytics/        # month-over-month trend analysis
├── 13_statistical_analysis/      # analysis-ready dataset for Python stats
├── 14_ml_features/               # point-in-time churn feature/label dataset
├── 15_reporting/                 # final reporting views
└── 16_validation/                # post-build validation checks
```

Every script is designed to be **rerunnable against a live database** without manual cleanup — tables are dropped and rebuilt deterministically.

## 🤖 Machine Learning — Customer Churn

**Business question:** *Among customers who have already shown engagement (2+ delivered orders), which ones are at risk of making no delivered purchase in the next 90 days?*

### Methodology highlights

- **Grain:** one row per customer-month prediction snapshot.
- **Population:** customers who have reached their second delivered order.
- **Snapshot logic:** monthly snapshots from the customer's second-order month through either their first observed 90-day churn event or the global observation cutoff (whichever comes first). Months with no purchase are retained so the model sees realistic inactivity gaps.
- **Target:** `churned = 1` when the customer has no delivered purchase in the following 90 days.
- **Leakage control:** every feature is computed strictly from activity observed **on or before** the snapshot month; only snapshots with a complete 90-day future observation window are kept; the final chronological period is held out as a true out-of-time test set.
- **Features:** 30/90/180-day rolling order counts and revenue, average order value, average review score, average delivery delay, active months, recency, recent order/revenue share, orders per active month, revenue per order, and recent-vs-older order ratio.

### Model comparison

Three model families are trained and compared with **expanding-window temporal cross-validation** and randomized hyperparameter search, optimizing primarily for **PR-AUC** (with ROC-AUC and F1 as secondary metrics):

| # | Model | Type | Why it's included |
|---|---|---|---|
| 1 | **Logistic Regression** | Linear baseline | Interpretable coefficients, fast, strong sanity-check baseline |
| 2 | **Random Forest** | Bagged tree ensemble | Captures nonlinearity and interactions with low tuning effort |
| 3 | **HistGradientBoosting** | Boosted trees | Typically strongest performer on tabular, mixed-scale numeric data |

```mermaid
flowchart LR
    A["Point-in-time<br/>churn dataset"] --> B["Chronological split<br/>train / validation / test"]
    B --> C["Randomized hyperparameter search<br/>+ expanding-window CV"]
    C --> D1[Logistic Regression]
    C --> D2[Random Forest]
    C --> D3[HistGradientBoosting]
    D1 --> E["Compare on PR-AUC<br/>(+ ROC-AUC, F1)"]
    D2 --> E
    D3 --> E
    E --> F["Select best pipeline"]
    F --> G[("models/best_churn_model.joblib")]
    C -.logged to.-> M[("MLflow<br/>Tracking Server")]

    style A fill:#4c72b0,color:#fff
    style G fill:#2ea44f,color:#fff
    style M fill:#0194E2,color:#fff
```

The performance-optimized build first materializes delivered orders into a compact `ml.delivered_orders` table, aggregates to `ml.customer_monthly_activity`, and only then constructs the point-in-time dataset — avoiding an expensive snapshot × full-order-history join and reducing the risk of Supabase statement timeouts on large SQL builds.

All experiment runs, hyperparameters, and metrics are tracked with **MLflow**. The selected best pipeline is persisted to `models/best_churn_model.joblib` and loaded directly by Streamlit — no separate model-serving API is required.

See [`docs/churn_methodology.md`](docs/churn_methodology.md), [`docs/ml_modeling.md`](docs/ml_modeling.md), and [`docs/model_card.md`](docs/model_card.md) for full details, including known limitations.

## 📊 Statistical Analysis

The SQL layer prepares analysis-ready data; Python (`scipy`) performs the formal statistical testing. Priority hypotheses investigated:

1. Delivery delay is associated with lower review scores.
2. Customer segments differ materially in purchase frequency and monetary value.
3. Retention differs meaningfully across acquisition cohorts.

Analysis follows good statistical practice: hypotheses are defined before inspecting results, tests are chosen based on the data-generating assumptions, effect sizes and uncertainty are reported (not just p-values), causal language is avoided where the design doesn't support it, and multiple-comparison correction is considered where relevant. See [`docs/statistical_analysis.md`](docs/statistical_analysis.md).

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Database | PostgreSQL (Supabase) |
| Data access | SQLAlchemy, psycopg 3 |
| Data wrangling | pandas, numpy, pyarrow |
| Statistics | scipy |
| Machine learning | scikit-learn |
| Experiment tracking | MLflow |
| Dashboard / app | Streamlit, Plotly, Matplotlib |
| Testing | pytest |
| Config | python-dotenv |

## 📁 Project Structure

```text
brazilian-ecommerce-data-science/
├── app/                      # Streamlit application
│   ├── streamlit_app.py      # entry point
│   ├── pages/                # executive, customers, retention, products,
│   │                          # sellers, delivery, churn
│   ├── components/           # theme, charts, data loaders, UI helpers
│   └── assets/                # styling
├── sql/                      # ordered, rerunnable SQL build (see above)
├── src/
│   ├── database/              # connection + I/O helpers
│   ├── features/              # churn feature engineering
│   ├── models/                # training, prediction, MLflow utilities
│   └── visualization/         # chart helpers
├── scripts/                  # CLI entry points (see below)
├── docs/                     # data model, methodology, business questions, model card
├── data/                     # generated dashboard + ML datasets (gitignored)
├── models/                   # trained model artifacts (gitignored)
├── mlflow/                   # local MLflow tracking store
├── tests/                    # pytest suite
└── requirements.txt
```

## 🚀 Getting Started

### 1. Clone and set up an environment

```bash
git clone https://github.com/Harshithpatali/brazilian-ecommerce-data-science.git
cd brazilian-ecommerce-data-science
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure database credentials

Copy `.env.example` to `.env` and provide either a full connection string:

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/postgres
```

or the individual Supabase variables:

```env
SUPABASE_DB_HOST=
SUPABASE_DB_PORT=5432
SUPABASE_DB_NAME=postgres
SUPABASE_DB_USER=
SUPABASE_DB_PASSWORD=
```

`.env` is gitignored — never commit real credentials.

### 4. Validate the connection

```bash
python scripts/check_connection.py
```

### 5. Build the full SQL analytical layer + ML dataset

The simplest path is the single orchestrator script, which runs every SQL folder in order:

```bash
python -m scripts.build_all
```

Or run folders individually if you want more control:

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

### 6. Build local dashboard + ML datasets

```bash
python scripts/build_dashboard_data.py   # writes data/dashboard/
python scripts/build_ml_data.py          # writes data/ml/customer_features.parquet
```

### 7. Train the churn model

```bash
python -m scripts.run_ml
```

This builds `data/ml/customer_churn_dataset.parquet`, trains and compares the candidate models with a chronological holdout, and writes:

- `models/best_churn_model.joblib` — selected fitted pipeline
- `models/model_comparison.csv` — benchmark across model families
- `models/model_metadata.json` — features, hyperparameters, split dates, test metrics
- `models/test_predictions.parquet` — final chronological test predictions

### 8. (Optional) Explore experiments in MLflow

```bash
python -m mlflow ui --backend-store-uri ./mlruns
```

Windows helper (avoids the `file://` URI pitfall on Windows paths):

```bash
python -m scripts.start_mlflow
```

### 9. Run the dashboard locally

```bash
streamlit run app/streamlit_app.py
```

The churn dashboard is available at the **Churn** page in the sidebar (`/churn`), and predicts the probability that a customer makes no delivered purchase in the next 90 days.

## ✅ Recommended End-to-End Workflow

```bash
python -m scripts.build_all
python -m scripts.validate_churn_data
python -m scripts.run_ml
python -m scripts.start_mlflow      # optional, local experiment UI
python -m streamlit run app/streamlit_app.py
```

The churn dataset is materialized in PostgreSQL using a compact customer-month table rather than a raw snapshot × full-order-history join — this avoids repeated expensive joins and reduces the chance of Supabase statement timeouts on large builds.

## 🧪 Testing

```bash
pytest
```

The test suite covers the database connection module and the churn feature-engineering contract (schema/shape guarantees the SQL layer must satisfy before modeling).

## 🔒 Security

`.env`, local database credentials, and generated model/data artifacts are excluded from version control via `.gitignore`. Use `.env.example` as the safe, credential-free template.

## 🎯 What This Project Demonstrates

- Advanced PostgreSQL/SQL — CTEs, window functions, `LATERAL` joins, careful grain management
- Data-quality profiling and referential-integrity validation
- Analytical data modeling (staging → analytics → ML layers)
- Customer segmentation (RFM) and cohort retention analysis
- Rigorous statistical hypothesis testing
- Leakage-safe, point-in-time feature engineering for time-aware ML
- Model comparison with temporal cross-validation and experiment tracking (MLflow)
- Production-style project structure: reproducible SQL builds, CLI scripts, tests, docs
- End-to-end delivery as a live, interactive business dashboard (Streamlit)

## 📄 Documentation

| Doc | Contents |
|---|---|
| [`docs/project_plan.md`](docs/project_plan.md) | Phase-by-phase build plan |
| [`docs/data_model.md`](docs/data_model.md) | Raw vs. analytical layer, customer identity rules |
| [`docs/business_questions.md`](docs/business_questions.md) | Business questions answered by each analytical layer |
| [`docs/statistical_analysis.md`](docs/statistical_analysis.md) | Hypotheses and statistical approach |
| [`docs/churn_methodology.md`](docs/churn_methodology.md) | Point-in-time churn label & feature methodology |
| [`docs/ml_modeling.md`](docs/ml_modeling.md) | Model families, tuning, saved artifacts |
| [`docs/model_card.md`](docs/model_card.md) | Model intended use and limitations |

## 📜 License

This project is released under the [MIT License](LICENSE). The underlying Olist dataset is subject to its own license on Kaggle — see the [dataset page](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) for terms.

## 🙋 Author

**Harshith Patali**
GitHub: [@Harshithpatali](https://github.com/Harshithpatali)

If you find this project useful or have suggestions, feel free to open an issue or a pull request.
