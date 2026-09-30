import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from common import load_segments, OFFICIAL_FEATURES, make_metrics
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd

df=load_segments()
train=df[df.train.eq(1)].copy()
test=df[df.train.eq(0)].copy()

rows=[]
for ch in sorted(test.channel.unique()):
    hold=test[test.channel.eq(ch)]
    if hold.anomaly.nunique()<2 or len(hold)<30:
        continue
    fit=train[~train.channel.eq(ch)]
    model=ExtraTreesClassifier(n_estimators=500,class_weight="balanced",random_state=42,n_jobs=-1)
    model.fit(fit[OFFICIAL_FEATURES],fit.anomaly)
    score=model.predict_proba(hold[OFFICIAL_FEATURES])[:,1]
    pred=(score>=0.5).astype(int)
    m=make_metrics(hold.anomaly,pred,score)
    m["Channel"]=ch; m["n"]=len(hold)
    rows.append(m)

out=pd.DataFrame(rows)
out.to_csv(Path(__file__).resolve().parents[1]/"results/channel_generalization_reproduced.csv",index=False)
print(out)
