import numpy as np
import pandas as pd
import pytest
from src.growth_engine.experiments import make_randomized_campaign_data
from src.growth_engine.uplift import fit_t_learner,uplift_at_k
from src.growth_engine.pricing import optimize_price
from src.growth_engine.fraud import make_demo_transactions,score_transactions
from src.growth_engine.decision_policy import recommend_action

def test_randomized_campaign_reproducible_and_balanced():
    a=make_randomized_campaign_data(1200,11); b=make_randomized_campaign_data(1200,11); pd.testing.assert_frame_equal(a,b); assert .4<a.treatment.mean()<.6

def test_t_learner_scores_holdout():
    r=fit_t_learner(make_randomized_campaign_data(1800,12)); assert len(r["scored"])>0; assert r["scored"].uplift.notna().all(); assert uplift_at_k(r["scored"],.2)["customers"]==int(np.ceil(len(r["scored"])*.2))

def test_price_optimizer_respects_cost_floor():
    assert optimize_price(-1.4,100,45,50,150)["recommended_price"]>=50

def test_anomaly_scores_and_labels():
    r=score_transactions(make_demo_transactions(300)); assert r.anomaly_score.between(0,100).all(); assert set(r.risk_level.unique())<={"Low","Medium","High"}

def test_high_risk_routes_to_manual_review():
    assert recommend_action(.2,90,92,100,1)["recommended_action"]=="Manual risk review"

def test_invalid_uplift_fraction_rejected():
    with pytest.raises(ValueError): uplift_at_k(pd.DataFrame({"uplift":[.1],"treatment":[1],"outcome":[1]}),0)
