import pandas as pd
import pytest

from src.customer_intelligence import build_rfm, campaign_scenario, kaplan_meier


@pytest.fixture
def sample():
    return pd.DataFrame({
        "customer_unique_id": ["a", "b", "c", "d"],
        "orders": [3, 1, 2, 1],
        "first_order_date": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
        "last_order_date": ["2020-03-01", "2020-02-01", "2020-04-01", "2020-04-01"],
        "second_order_date": ["2020-02-01", None, "2020-04-01", None],
        "revenue": [300, 50, 150, 20],
        "aov": [100, 50, 75, 20],
    })


def test_rfm_has_explanations_and_valid_scores(sample):
    result = build_rfm(sample, "2020-05-01")
    assert result[["R_score", "F_score", "M_score"]].isin([1, 2, 3, 4, 5]).all().all()
    assert result["customer_explanation"].str.len().gt(20).all()
    assert set(result["rfm_segment"]).issubset({"Champions", "New / Promising", "At Risk", "Hibernating", "Loyal / Developing"})


def test_kaplan_meier_is_monotonic_and_bounded(sample):
    result = kaplan_meier(sample)
    assert result["survival"].between(0, 1).all()
    assert result["survival"].is_monotonic_decreasing
    assert result["repeat_purchase_cumulative"].between(0, 1).all()


def test_campaign_scenario_is_assumption_driven(sample):
    rfm = build_rfm(sample, "2020-05-01")
    result = campaign_scenario(rfm, "All customers", 0.5, 2.0, 0.5, 0.3)
    assert result["contacted_customers"] == 2
    assert result["assumed_incremental_repeaters"] == pytest.approx(0.04)
    assert result["scenario_net_contribution"] == pytest.approx(result["gross_contribution"] - result["campaign_cost"])
