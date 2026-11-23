def recommend_action(uplift: float,churn_risk: float,anomaly_score: float,expected_margin: float,contact_cost: float,discount_cost: float=0.) -> dict:
    """Transparent rules-based triage; not a validated production policy."""
    if min(churn_risk,anomaly_score)<0 or max(churn_risk,anomaly_score)>100: raise ValueError("Risk scores must be between 0 and 100")
    if contact_cost<0 or discount_cost<0: raise ValueError("Costs cannot be negative")
    if anomaly_score>=80: action,reason="Manual risk review","High anomaly score; pause automated incentives pending review."
    elif uplift>0 and expected_margin*uplift>contact_cost+discount_cost: action,reason="Target with controlled offer","Estimated incremental contribution exceeds intervention cost."
    elif churn_risk>=70: action,reason="Low-cost retention contact","High inactivity risk without a validated profitable discount recommendation."
    else: action,reason="No incentive","Insufficient estimated incremental value for a paid intervention."
    return {"recommended_action":action,"reason":reason,"estimated_net_value":float(expected_margin*uplift-contact_cost-discount_cost)}
