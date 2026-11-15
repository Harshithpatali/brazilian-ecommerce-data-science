import streamlit as st
from src.advanced_decision_science import two_proportion_power, decision_roi, sequential_looks


def render():
    st.title("Advanced Decision Science")
    st.caption("Experiment planning and observational logistics decision support. All causal and financial assumptions must be validated.")
    tabs=st.tabs(["Experiment power","Logistics economics","Causal study design","CLV and survival"])
    with tabs[0]:
        baseline=st.number_input("Baseline repeat rate",.001,.95,.03,.005,format="%.3f")
        mde=st.number_input("Minimum detectable absolute lift",.001,.5,.01,.001,format="%.3f")
        alpha=st.select_slider("Alpha",[.01,.025,.05,.10],value=.05)
        power=st.select_slider("Power",[.7,.8,.85,.9,.95],value=.8)
        if baseline+mde<1:
            r=two_proportion_power(baseline,mde,alpha,power)
            c1,c2=st.columns(2); c1.metric("Per arm",f"{r['per_arm']:,}"); c2.metric("Total sample",f"{r['total']:,}")
        st.markdown("A/A tests and pre-registered stopping rules are required. Repeated uncorrected peeking inflates false positives.")
        raw=st.text_input("Sequential p-values","0.20,0.04,0.03")
        if st.button("Evaluate sequential looks"):
            st.dataframe(sequential_looks([float(x) for x in raw.split(",") if x.strip()]),use_container_width=True)
    with tabs[1]:
        effect=st.number_input("Estimated repeat-rate effect (absolute)",value=.01,step=.001,format="%.3f")
        aov=st.number_input("Expected repeat order value (R$)",min_value=0.,value=180.)
        margin=st.slider("Contribution margin",0,100,25)/100
        customers=st.number_input("Customers affected",min_value=1,value=1000)
        cost=st.number_input("Logistics investment (R$)",min_value=0.,value=1000.)
        r=decision_roi(effect,aov,margin,int(customers),cost)
        st.metric("Net contribution",f"R$ {r['net_value_brl']:,.2f}"); st.write(r)
        st.warning("Use only an identified causal effect or randomized experiment estimate, not the raw late/on-time difference.")
    with tabs[2]:
        st.markdown("""
**Treatment:** late delivery, defined as actual delivery after the estimate.

**Outcome:** second delivered purchase within a fixed window after the index delivery.

**Pre-treatment confounders:** product category, basket value, freight, seller/customer state, purchase month and product weight/dimensions.

**Do not adjust for review score** in the total-effect model: it occurs after delivery and may mediate the effect.

Required diagnostics: propensity overlap, weighted standardized mean differences, effective sample size, cross-fitted doubly robust estimate, confidence interval and sensitivity to unmeasured confounding.

Olist is observational. Identification requires conditional exchangeability, positivity and consistency.
""")
    with tabs[3]:
        st.markdown("""
- Fit BG/NBD on calibration transactions and compare predicted versus observed purchases by frequency bucket.
- Fit Gamma-Gamma only on repeat customers; report the small fitting cohort and parameter instability.
- Keep the existing monthly 90-day propensity model separate from the new first-order 180-day target view.
- Extend existing Kaplan-Meier with Cox PH diagnostics; use stratification or Weibull AFT if proportional hazards fails.
- Discuss cure models because a substantial fraction may never return; ordinary survival models do not identify a permanent cure fraction by default.
""")
