import numpy as np
import pandas as pd
import pytest
from src.growth_engine.campaign_optimizer import allocate_campaign_budget
from src.growth_engine.monitoring import population_stability_index,drift_status

def test_campaign_budget_respected_and_negative_value_excluded():
    data=pd.DataFrame({"customer_id":["a","b","c"],"uplift":[.3,.2,-.1]})
    result=allocate_campaign_budget(data,10,2,100,1)
    assert result["spend"]<=10
    assert result["customers_selected"]==3
    assert result["expected_net_value"]>0

def test_psi_detects_distribution_shift():
    rng=np.random.default_rng(3); ref=rng.normal(0,1,1000); current=rng.normal(2,1,1000)
    psi=population_stability_index(ref,current)
    assert psi>0.1
    assert drift_status(psi) in {"Monitor","Investigate"}

def test_psi_requires_sufficient_data():
    with pytest.raises(ValueError): population_stability_index([1,2],[1,2])
