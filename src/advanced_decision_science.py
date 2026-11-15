"""Olist research utilities: first-order targets, observational AIPW and experiments."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def first_order_repeat_target(orders, horizon_days=180, observation_end=None):
    required={"customer_unique_id","order_purchase_timestamp","order_status"}
    if required-set(orders): raise ValueError(f"Missing columns: {sorted(required-set(orders))}")
    d=orders.loc[orders.order_status.eq("delivered")].copy()
    d["order_purchase_timestamp"]=pd.to_datetime(d.order_purchase_timestamp,utc=True)
    d=d.dropna(subset=["customer_unique_id","order_purchase_timestamp"]).sort_values(["customer_unique_id","order_purchase_timestamp"])
    first=d.drop_duplicates("customer_unique_id").copy()
    end=pd.Timestamp(observation_end) if observation_end is not None else d.order_purchase_timestamp.max()
    end=end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    first=first.loc[first.order_purchase_timestamp+pd.Timedelta(days=horizon_days)<=end].copy()
    first["repeat_within_horizon"]=0
    pairs=d.merge(first[["customer_unique_id","order_purchase_timestamp"]].rename(columns={"order_purchase_timestamp":"first_ts"}),on="customer_unique_id")
    hit=pairs.loc[(pairs.order_purchase_timestamp>pairs.first_ts)&(pairs.order_purchase_timestamp<=pairs.first_ts+pd.Timedelta(days=horizon_days)),"customer_unique_id"].unique()
    first["repeat_within_horizon"]=first.customer_unique_id.isin(hit).astype("int8")
    return first.reset_index(drop=True)


def fit_bgnbd_gamma_gamma(transactions, observation_end):
    try:
        from lifetimes import BetaGeoFitter, GammaGammaFitter
        from lifetimes.utils import summary_data_from_transaction_data
    except ImportError as exc:
        raise ImportError("Install lifetimes to fit BG/NBD and Gamma-Gamma.") from exc
    summary=summary_data_from_transaction_data(transactions,"customer_unique_id","date",monetary_value_col="monetary_value",observation_period_end=pd.Timestamp(observation_end),freq="D")
    bg=BetaGeoFitter(penalizer_coef=.01).fit(summary.frequency,summary.recency,summary.T)
    repeaters=summary.loc[summary.frequency>0]
    gg=GammaGammaFitter(penalizer_coef=.01).fit(repeaters.frequency,repeaters.monetary_value) if len(repeaters)>=10 else None
    return {"summary":summary,"bgnbd":bg,"gamma_gamma":gg,"repeaters":len(repeaters),"warning":"Sparse repeat histories make parameters weakly identified."}


def crossfit_dr_effect(X,treatment,outcome,n_splits=5,random_state=42):
    X=pd.get_dummies(X,dummy_na=True,dtype=float).replace([np.inf,-np.inf],np.nan).fillna(0)
    a=np.asarray(treatment,dtype=int); y=np.asarray(outcome,dtype=float)
    if set(np.unique(a))!={0,1}: raise ValueError("Treatment must contain both arms.")
    e=np.zeros(len(y)); m0=np.zeros(len(y)); m1=np.zeros(len(y))
    cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=random_state)
    for tr,te in cv.split(X,a):
        ps=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(X.iloc[tr],a[tr])
        e[te]=ps.predict_proba(X.iloc[te])[:,1]
        for arm,out in ((0,m0),(1,m1)):
            idx=tr[a[tr]==arm]
            model=RandomForestRegressor(n_estimators=150,min_samples_leaf=10,n_jobs=-1,random_state=random_state).fit(X.iloc[idx],y[idx])
            out[te]=model.predict(X.iloc[te])
    ec=np.clip(e,.01,.99)
    score=m1-m0+a*(y-m1)/ec-(1-a)*(y-m0)/(1-ec)
    smd={}
    for col in X:
        v=X[col].to_numpy(); pooled=np.sqrt((np.var(v[a==1])+np.var(v[a==0]))/2)
        smd[col]=float((v[a==1].mean()-v[a==0].mean())/pooled) if pooled else 0.
    se=float(np.std(score,ddof=1)/np.sqrt(len(y)))
    return {"ate":float(score.mean()),"standard_error":se,"ci_95":[float(score.mean()-1.96*se),float(score.mean()+1.96*se)],"propensity":e,"overlap_fraction":float(np.mean((e>=.01)&(e<=.99))),"max_abs_smd_before":max(map(abs,smd.values()),default=0.),"smd_before":smd,"interpretation":"Observational AIPW; requires no unmeasured confounding, positivity and consistency."}


def ipw_weights(treatment,propensity):
    a=np.asarray(treatment,dtype=int); p=np.clip(np.asarray(propensity,dtype=float),.01,.99)
    w=a/p+(1-a)/(1-p)
    return w/w.mean()


def two_proportion_power(baseline_rate,mde,alpha=.05,power=.8):
    p1=baseline_rate; p2=p1+mde
    if not 0<p1<p2<1: raise ValueError("Require 0 < baseline < baseline + MDE < 1.")
    za=stats.norm.ppf(1-alpha/2); zp=stats.norm.ppf(power); pooled=(p1+p2)/2
    n=(za*np.sqrt(2*pooled*(1-pooled))+zp*np.sqrt(p1*(1-p1)+p2*(1-p2)))**2/(p2-p1)**2
    return {"per_arm":int(np.ceil(n)),"total":2*int(np.ceil(n)),"baseline_rate":p1,"treatment_rate":p2,"mde":mde,"alpha":alpha,"power":power}


def cuped_adjust(outcome,pre_period):
    y=np.asarray(outcome,dtype=float); x=np.asarray(pre_period,dtype=float)
    if len(y)!=len(x) or len(y)<3: raise ValueError("Equal-length arrays with at least 3 rows required.")
    theta=np.cov(y,x,ddof=1)[0,1]/np.var(x,ddof=1); adj=y-theta*(x-x.mean())
    return {"theta":float(theta),"adjusted_outcome":adj,"variance_reduction":float(1-np.var(adj,ddof=1)/np.var(y,ddof=1))}


def bootstrap_mean_difference(control,treatment,n_boot=2000,random_state=42):
    c=np.asarray(control,dtype=float); t=np.asarray(treatment,dtype=float); rng=np.random.default_rng(random_state)
    draws=np.array([rng.choice(t,len(t),True).mean()-rng.choice(c,len(c),True).mean() for _ in range(n_boot)])
    return {"difference":float(t.mean()-c.mean()),"ci_95":[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]}


def sequential_looks(p_values,alpha=.05):
    p=np.asarray(p_values,dtype=float); k=len(p); threshold=alpha/k
    return pd.DataFrame({"look":np.arange(1,k+1),"p_value":p,"alpha_spent":threshold,"reject":p<threshold})


def decision_roi(repeat_rate_effect,repeat_order_value,contribution_margin,customers,logistics_cost=0):
    orders=repeat_rate_effect*customers; revenue=orders*repeat_order_value; contribution=revenue*contribution_margin
    return {"incremental_repeat_orders":float(orders),"incremental_revenue_brl":float(revenue),"incremental_contribution_brl":float(contribution),"logistics_cost_brl":float(logistics_cost),"net_value_brl":float(contribution-logistics_cost)}


def sensitivity_e_value(risk_ratio):
    rr=max(float(risk_ratio),1/float(risk_ratio))
    return rr+np.sqrt(rr*(rr-1))
