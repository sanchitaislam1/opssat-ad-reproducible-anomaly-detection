import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import load_segments, OFFICIAL_FEATURES

def safe_entropy(p):
    p=np.asarray(p,float)
    p=p[p>0]
    if len(p)==0: return 0.0
    p=p/p.sum()
    return float(-(p*np.log(p+1e-12)).sum()/np.log(len(p))) if len(p)>1 else 0.0

def raw_features(g):
    # Identify the signal/timestamp columns from the official raw file.
    numeric=[c for c in g.columns if pd.api.types.is_numeric_dtype(g[c])]
    signal_col=next((c for c in numeric if c not in {"segment","anomaly","train"} and "time" not in c.lower()), None)
    time_col=next((c for c in g.columns if "time" in c.lower() or "timestamp" in c.lower()), None)
    if signal_col is None:
        raise ValueError("Could not infer raw signal column; inspect segments.csv and set signal_col explicitly.")
    x=g[signal_col].astype(float).to_numpy()
    x=x[np.isfinite(x)]
    if len(x)<4: return pd.Series(dtype=float)
    d=np.diff(x)
    med=np.median(x)
    mad=np.median(np.abs(x-med))
    q25,q75=np.percentile(x,[25,75])
    iqr=q75-q25
    zc=np.mean((x[:-1]*x[1:])<0)
    ac=float(np.corrcoef(x[:-1],x[1:])[0,1]) if np.std(x)>0 else 0.0
    rough=float(np.mean(np.abs(d))/(np.std(x)+1e-12))
    freqs,powv=periodogram(x)
    powv=np.maximum(powv,0)
    total=powv.sum()+1e-12
    idx=int(np.argmax(powv[1:])+1) if len(powv)>1 else 0
    low=powv[1:int(len(powv)*0.25)].sum()/total
    high=powv[int(len(powv)*0.75):].sum()/total
    return pd.Series({
        "median":med,"iqr":iqr,"mad":mad,"range":float(x.max()-x.min()),
        "robust_range":float((q75-q25)/(mad+1e-12)),
        "zero_crossing_rate":zc,"lag1_autocorr":ac,
        "mean_abs_successive_diff":float(np.mean(np.abs(d))),
        "normalized_roughness":rough,
        "spectral_entropy":safe_entropy(powv),
        "dominant_frequency_position":float(freqs[idx]/(freqs.max()+1e-12)) if len(freqs)>0 else 0,
        "low_frequency_power":low,"high_frequency_power":high,
        "high_low_power_ratio":float(high/(low+1e-12)),
    })

df=load_segments()
# This script assumes one raw signal column per segment grouping. If your release uses a different
# column naming convention, set signal_col/time_col explicitly above.
groups=[]
for seg,g in df.groupby("segment",sort=False):
    r=raw_features(g)
    if len(r):
        r["segment"]=seg
        groups.append(r)
raw=pd.DataFrame(groups)

base=df.drop_duplicates("segment")[["segment","train","anomaly"]+OFFICIAL_FEATURES]
data=base.merge(raw,on="segment",how="inner")
new=[c for c in raw.columns if c!="segment"]
print("Generated",len(new),"additional features:",new)
data.to_csv(Path(__file__).resolve().parents[1]/"results/expanded_features_generated.csv",index=False)

train=data[data.train.eq(1)]
X=train[OFFICIAL_FEATURES+new].replace([np.inf,-np.inf],np.nan).fillna(0)
y=train.anomaly.astype(int)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
for label,cols in {
    "official":OFFICIAL_FEATURES,
    "robust":OFFICIAL_FEATURES+new[:5],
    "temporal":OFFICIAL_FEATURES+new[5:9],
    "spectral":OFFICIAL_FEATURES+new[9:],
    "all":OFFICIAL_FEATURES+new
}.items():
    scores=cross_val_score(
        RandomForestClassifier(n_estimators=500,class_weight="balanced",random_state=42,n_jobs=-1),
        X[cols],y,cv=cv,scoring="f1",n_jobs=-1)
    print(label, scores.mean(), scores.std())
