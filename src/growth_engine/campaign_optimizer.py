from __future__ import annotations
import numpy as np
import pandas as pd

def allocate_campaign_budget(scored: pd.DataFrame,budget: float,contact_cost: float,contribution_per_conversion: float,discount_cost: float=0.,max_customers: int|None=None) -> dict:
    """Greedy positive-net-value targeting under a fixed campaign budget.

    Expected value uses model-estimated uplift and is a scenario until uplift is
    validated in a randomized experiment. Each contacted customer has equal cost.
    """
    required={"customer_id","uplift"}
    missing=required.difference(scored.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    if budget<0 or contact_cost<0 or discount_cost<0 or contribution_per_conversion<0: raise ValueError("Budget and unit economics must be non-negative")
    d=scored.copy()
    d["expected_incremental_value"]=d.uplift.clip(lower=0)*contribution_per_conversion-contact_cost-discount_cost
    d=d.loc[d.expected_incremental_value>0].sort_values("expected_incremental_value",ascending=False)
    capacity=int(budget//max(contact_cost+discount_cost,1e-12)) if contact_cost+discount_cost>0 else len(d)
    if max_customers is not None: capacity=min(capacity,int(max_customers))
    selected=d.head(max(0,capacity)).copy()
    spend=len(selected)*(contact_cost+discount_cost)
    return {"selected":selected,"customers_selected":len(selected),"budget":float(budget),"spend":float(spend),"expected_incremental_value":float(selected.expected_incremental_value.sum()),"expected_net_value":float(selected.expected_incremental_value.sum()),"budget_remaining":float(budget-spend),"interpretation":"Expected value is model-based scenario output, not measured campaign profit."}
