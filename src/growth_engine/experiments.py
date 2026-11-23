from __future__ import annotations
import numpy as np
import pandas as pd

def make_randomized_campaign_data(n: int = 12000, seed: int = 42) -> pd.DataFrame:
    """Reproducible randomized campaign benchmark; not observed Olist campaign data."""
    if n < 500: raise ValueError("n must be at least 500")
    rng=np.random.default_rng(seed)
    recency=rng.gamma(2.0,35.0,n).clip(1,365); frequency=rng.poisson(2.0,n).clip(0,15)
    monetary=rng.lognormal(4.2,.8,n).clip(5,2000); affinity=rng.beta(2,3,n); sensitivity=rng.beta(2,2,n)
    X=pd.DataFrame({"recency_days":recency,"purchase_frequency":frequency,"customer_value":monetary,"category_affinity":affinity,"discount_sensitivity":sensitivity})
    treatment=rng.binomial(1,.5,n)
    base_logit=-2-.003*recency+.12*frequency+.00035*monetary+.7*affinity
    base=1/(1+np.exp(-base_logit)); tau=.015+.12*sensitivity+.035*affinity-.00004*np.minimum(monetary,1000)
    treated=np.clip(base+tau,.005,.85); outcome=rng.binomial(1,np.where(treatment==1,treated,base))
    return X.assign(treatment=treatment,outcome=outcome,true_uplift=tau,base_conversion_probability=base)
