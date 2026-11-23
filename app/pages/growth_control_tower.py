from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app.components.ui import hero,metric_card,section_title
from src.growth_engine.experiments import make_randomized_campaign_data
from src.growth_engine.uplift import fit_t_learner,uplift_at_k
from src.growth_engine.fraud import make_demo_transactions,score_transactions

BLUE="#3b82f6"; ORANGE="#f97316"; PURPLE="#9333ea"; PANEL="#171333"

def chart(fig,height=250):
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor=PANEL,font=dict(color="#e7e4f5",size=10),margin=dict(l=8,r=8,t=30,b=8),height=height,legend=dict(orientation="h",y=1.08,x=0))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

def render():
    st.markdown("<div class='control-tower-kicker'>COMMERCE OPERATIONS · GROWTH · RISK</div>",unsafe_allow_html=True)
    hero("E-commerce Growth & Risk Control Tower","A unified view of customer growth, returns, payments and transaction risk.")
    st.caption("DEMO MODE · Operational return/payment events and fraud signals are synthetic examples. Campaign uplift is evaluated on a randomized simulation, not Olist campaign history.")
    raw=make_demo_transactions(); risk=score_transactions(raw); exp=make_randomized_campaign_data(6000); uplift=fit_t_learner(exp); top20=uplift_at_k(uplift["scored"],.2)
    rng=np.random.default_rng(24); days=np.arange(1,31); trend=pd.DataFrame({"Day":days,"Orders":np.maximum(25,52+12*np.sin(days/2.4)+rng.normal(0,6,30)).round().astype(int)})
    cols=st.columns(5)
    for col,(label,value,accent) in zip(cols,[("Total payments","4,753","blue"),("COD orders","904","violet"),("Online orders","936","teal"),("Payment failures","157","rose"),("Refunds processed","624","amber")]):
        with col: metric_card(label,value,accent=accent)
    a,b,c=st.columns([1.55,1,1])
    with a:
        section_title("Orders trend overview"); fig=px.line(trend,x="Day",y="Orders",markers=True); fig.update_traces(line_color=BLUE,marker_color=BLUE,line_width=2); chart(fig)
    with b:
        section_title("Cancellation reasons"); fig=go.Figure(go.Pie(labels=["Out of stock","Payment failed","Customer cancelled","Address issue"],values=[410,95,62,28],hole=.62,marker=dict(colors=[BLUE,ORANGE,PURPLE,"#c084fc"]))); fig.update_layout(showlegend=True,annotations=[dict(text="595<br>Total",x=.5,y=.5,showarrow=False,font_size=15)]); chart(fig)
    with c:
        section_title("Return reasons"); fig=go.Figure(go.Pie(labels=["Quality issue","Not as described","Changed mind","Size/fit issue","Damaged product"],values=[410,82,58,43,31],hole=.62,marker=dict(colors=[BLUE,ORANGE,PURPLE,"#c084fc","#64748b"]))); fig.update_layout(showlegend=True,annotations=[dict(text="624<br>Total",x=.5,y=.5,showarrow=False,font_size=15)]); chart(fig)
    left,right=st.columns([1.35,1])
    with left:
        section_title("High-risk customers · potential anomalies")
        table=risk.loc[risk.risk_level.eq("High"),["customer_id","order_value","return_count","refund_ratio","anomaly_score","risk_level"]].head(12).copy()
        table.columns=["Customer ID","Order value","Returns","Refund ratio","Risk score","Risk level"]
        st.dataframe(table.style.format({"Order value":"R$ {:,.2f}","Refund ratio":"{:.1%}","Risk score":"{:.0f}"}),use_container_width=True,hide_index=True,height=260)
    with right:
        section_title("Fraud score distribution"); counts=risk.risk_level.value_counts().reindex(["High","Medium","Low"]).fillna(0).astype(int)
        fig=go.Figure(go.Pie(labels=counts.index,values=counts.values,hole=.6,marker=dict(colors=[ORANGE,"#a855f7",BLUE]))); fig.update_layout(annotations=[dict(text=f"{len(risk):,}<br>Customers",x=.5,y=.5,showarrow=False,font_size=14)]); chart(fig,260)
    left,right=st.columns(2)
    with left:
        section_title("Fraud score trend"); daily=risk.groupby("event_day").anomaly_score.mean().reset_index(name="Average risk score"); fig=px.line(daily,x="event_day",y="Average risk score",markers=True); fig.update_traces(line_color=BLUE,line_width=2); chart(fig,230)
    with right:
        section_title("Growth campaign · uplift benchmark")
        summary=pd.DataFrame({"Metric":["Randomized experiment ITT","T-learner estimated ATE","Top-20% estimated uplift"],"Value":[uplift["observed_itt"],uplift["ate_model_estimate"],top20["uplift"]]})
        fig=px.bar(summary,x="Metric",y="Value",color="Metric",color_discrete_sequence=[BLUE,PURPLE,ORANGE]); fig.update_layout(showlegend=False,yaxis_tickformat=".1%"); chart(fig,230)
    section_title("Recent alerts")
    for col,(title,body,kind) in zip(st.columns(5),[("High return rate detected","Synthetic customer with elevated returns","warning"),("Multiple returns for same product","Review repeat-return pattern","warning"),("High refund amount","Refund exceeds review threshold","danger"),("COD order pattern suspicious","Review payment behavior","warning"),("Potential fraud detected","Anomaly score ≥ 80 · manual review","danger")]):
        with col: st.markdown(f"<div class='alert-tile {kind}'><b>{title}</b><span>{body}</span></div>",unsafe_allow_html=True)
    with st.expander("Model details and downloadable review queue"):
        st.markdown("**Uplift:** T-learner trained and evaluated on a randomized synthetic campaign benchmark. Not measured Olist campaign lift.")
        st.write({"Benchmark rows":len(exp),"Holdout rows":len(uplift["scored"]),"Randomized ITT":uplift["observed_itt"],"Model estimated ATE":uplift["ate_model_estimate"],"Top-20% uplift":top20["uplift"]})
        st.markdown("**Fraud:** Isolation Forest anomaly ranking only. It does not produce confirmed fraud labels or calibrated fraud probabilities.")
        st.download_button("Download synthetic risk review queue",risk.to_csv(index=False).encode("utf-8"),"synthetic_risk_review_queue.csv","text/csv")
