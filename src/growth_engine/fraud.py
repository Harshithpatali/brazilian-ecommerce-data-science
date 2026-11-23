from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
FRAUD_FEATURES=["order_value","refund_ratio","orders_24h","payment_failures","return_count","account_age_days"]

def score_transactions(data: pd.DataFrame,contamination: float=.04,random_state: int=42) -> pd.DataFrame:
    """Unsupervised anomaly ranking; a high score is not proof of fraud."""
    missing=set(FRAUD_FEATURES).difference(data.columns)
    if missing: raise ValueError(f"Missing transaction features: {sorted(missing)}")
    if not 0<contamination<.5: raise ValueError("contamination must be between 0 and .5")
    X=data[FRAUD_FEATURES].replace([np.inf,-np.inf],np.nan); X=X.fillna(X.median()).fillna(0)
    model=IsolationForest(n_estimators=240,contamination=contamination,random_state=random_state,n_jobs=-1).fit(X)
    anomaly=-model.score_samples(X); lo,hi=float(anomaly.min()),float(anomaly.max()); score=100*(anomaly-lo)/(hi-lo) if hi>lo else np.zeros(len(anomaly))
    out=data.copy(); out["anomaly_score"]=score; out["risk_level"]=pd.cut(score,[-.001,60,80,100.001],labels=["Low","Medium","High"]).astype(str); out["review_recommended"]=model.predict(X)==-1
    return out.sort_values("anomaly_score",ascending=False).reset_index(drop=True)

def make_demo_transactions(n: int=1800,seed: int=7) -> pd.DataFrame:
    """Synthetic operational records; not real Olist fraud labels or customer activity."""
    rng=np.random.default_rng(seed); s=rng.binomial(1,.07,n)
    return pd.DataFrame({"customer_id":[f"CUST{i:06d}" for i in range(n)],"order_value":rng.lognormal(4.1+s*.75,.65,n),"refund_ratio":np.clip(rng.beta(1.2+s*5,10,n),0,1),"orders_24h":rng.poisson(1+s*3,n),"payment_failures":rng.poisson(.15+s*1.8,n),"return_count":rng.poisson(.25+s*2,n),"account_age_days":rng.integers(1,1600,n)*(1-s)+rng.integers(1,45,n)*s,"event_day":rng.integers(1,31,n)})
