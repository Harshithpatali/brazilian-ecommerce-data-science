import numpy as np
import pandas as pd
import pytest
from src.advanced_decision_science import first_order_repeat_target,two_proportion_power,cuped_adjust,sequential_looks,decision_roi,sensitivity_e_value


def test_first_order_target_uses_customer_unique_id_and_first_order_features():
 d=pd.DataFrame({"customer_unique_id":["a","a","b"],"order_purchase_timestamp":["2020-01-01","2020-02-01","2020-01-01"],"order_status":["delivered"]*3,"order_id":["a1","a2","b1"],"basket":[10,99,20]})
 x=first_order_repeat_target(d,180,"2021-01-01").set_index("customer_unique_id")
 assert x.loc["a","basket"]==10 and x.loc["a","repeat_within_horizon"]==1


def test_power_increases_for_smaller_effect():
 assert two_proportion_power(.03,.01)["total"]>two_proportion_power(.03,.02)["total"]


def test_cuped_reduces_correlated_noise():
 x=np.arange(100,dtype=float); y=x+np.random.default_rng(2).normal(0,1,100)
 assert cuped_adjust(y,x)["variance_reduction"]>.8


def test_sequential_guard_and_roi():
 assert not sequential_looks([.04,.04,.04]).reject.any()
 assert decision_roi(.02,100,.25,1000,200)["net_value_brl"]==300
 assert sensitivity_e_value(1.5)>1.5


def test_invalid_power_input():
 with pytest.raises(ValueError): two_proportion_power(.03,.97)
