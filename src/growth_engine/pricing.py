from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def fit_demand_elasticity(data: pd.DataFrame) -> dict:
    """Descriptive log-log price elasticity; not automatically causal."""
    if {"price","units_sold"}.difference(data.columns): raise ValueError("price and units_sold are required")
    d=data.loc[(data.price>0)&(data.units_sold>0)].copy()
    if len(d)<30: raise ValueError("At least 30 positive observations are required")
    X=np.log(d[["price"]]); y=np.log(d.units_sold); model=LinearRegression().fit(X,y)
    return {"elasticity":float(model.coef_[0]),"r_squared":float(model.score(X,y)),"n_observations":len(d),"model":model,"warning":"Observational association; price endogeneity and confounding prevent a causal interpretation without stronger design."}

def optimize_price(elasticity: float, reference_price: float, unit_cost: float,min_price: float|None=None,max_price: float|None=None,candidates: int=101) -> dict:
    if reference_price<=0 or unit_cost<0 or candidates<3: raise ValueError("Invalid price, cost or candidate count")
    floor=max(float(min_price if min_price is not None else unit_cost),unit_cost); ceiling=float(max_price if max_price is not None else reference_price*1.25)
    if ceiling<=floor: raise ValueError("Maximum price must exceed the cost-constrained floor")
    prices=np.linspace(floor,ceiling,candidates); demand=(prices/reference_price)**float(elasticity); contribution=(prices-unit_cost)*demand; i=int(np.argmax(contribution))
    return {"recommended_price":float(prices[i]),"reference_price":float(reference_price),"unit_cost":float(unit_cost),"relative_demand_index":float(demand[i]),"relative_revenue_index":float(prices[i]*demand[i]/reference_price),"relative_contribution_index":float(contribution[i]),"candidate_prices":prices,"contribution_curve":contribution}
