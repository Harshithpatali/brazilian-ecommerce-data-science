from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
FEATURES=["recency_days","purchase_frequency","customer_value","category_affinity","discount_sensitivity"]

def fit_t_learner(data: pd.DataFrame, random_state: int=42) -> dict:
    """Fit separate treatment/control response models and score a held-out set."""
    required=set(FEATURES+["treatment","outcome"]); missing=required.difference(data.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    if set(data.treatment.dropna().unique())!={0,1}: raise ValueError("Both treatment arms are required")
    train,test=train_test_split(data,test_size=.25,random_state=random_state,stratify=data.treatment); models={}
    for arm in (0,1):
        arm_data=train.loc[train.treatment==arm]
        if arm_data.outcome.nunique()<2: raise ValueError("Each arm must contain both outcome classes")
        model=RandomForestClassifier(n_estimators=180,min_samples_leaf=15,max_features=.9,n_jobs=-1,random_state=random_state+arm)
        model.fit(arm_data[FEATURES],arm_data.outcome); models[arm]=model
    scored=test.copy(); scored["p_control"]=models[0].predict_proba(test[FEATURES])[:,1]; scored["p_treatment"]=models[1].predict_proba(test[FEATURES])[:,1]; scored["uplift"]=scored.p_treatment-scored.p_control
    itt=float(test.loc[test.treatment==1,"outcome"].mean()-test.loc[test.treatment==0,"outcome"].mean())
    return {"models":models,"scored":scored,"features":FEATURES,"ate_model_estimate":float(scored.uplift.mean()),"observed_itt":itt}

def uplift_at_k(scored: pd.DataFrame, fraction: float=.2) -> dict:
    if not 0<fraction<=1: raise ValueError("fraction must be in (0, 1]")
    d=scored.sort_values("uplift",ascending=False).head(max(1,int(np.ceil(len(scored)*fraction))))
    t=d.loc[d.treatment==1,"outcome"]; c=d.loc[d.treatment==0,"outcome"]
    return {"customers":len(d),"treated":len(t),"control":len(c),"uplift":float(t.mean()-c.mean()) if len(t) and len(c) else float("nan")}

def uplift_curve(scored: pd.DataFrame) -> pd.DataFrame:
    d=scored.sort_values("uplift",ascending=False).reset_index(drop=True); rows=[]
    for k in range(1,len(d)+1):
        part=d.iloc[:k]; t=part.loc[part.treatment==1,"outcome"]; c=part.loc[part.treatment==0,"outcome"]
        rows.append({"customers_ranked":k,"fraction_targeted":k/len(d),"incremental_rate":float(t.mean()-c.mean()) if len(t) and len(c) else np.nan})
    return pd.DataFrame(rows)
