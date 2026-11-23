from __future__ import annotations
import numpy as np
import pandas as pd

def population_stability_index(reference, current, bins: int=10) -> float:
    """Calculate PSI for a numeric feature using reference-population quantile bins."""
    ref=pd.to_numeric(pd.Series(reference),errors="coerce").dropna().to_numpy(); cur=pd.to_numeric(pd.Series(current),errors="coerce").dropna().to_numpy()
    if len(ref)<20 or len(cur)<20: raise ValueError("At least 20 valid observations are required in each sample")
    edges=np.unique(np.quantile(ref,np.linspace(0,1,bins+1)))
    if len(edges)<3: return 0.0
    edges[0],edges[-1]=-np.inf,np.inf
    r=np.histogram(ref,bins=edges)[0]/len(ref); c=np.histogram(cur,bins=edges)[0]/len(cur)
    r=np.clip(r,1e-6,None); c=np.clip(c,1e-6,None)
    return float(np.sum((c-r)*np.log(c/r)))

def drift_status(psi: float) -> str:
    if psi<.1: return "Stable"
    if psi<.25: return "Monitor"
    return "Investigate"
