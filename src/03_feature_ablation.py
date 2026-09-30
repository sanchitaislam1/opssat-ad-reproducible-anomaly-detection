import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from common import load_segments, split_xy, FEATURE_GROUPS, OFFICIAL_FEATURES, make_metrics
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

df=load_segments()
Xtr, Xte, ytr, yte=split_xy(df)

rows=[]
for group, cols in FEATURE_GROUPS.items():
    model=ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(Xtr[cols], ytr)
    score=model.predict_proba(Xte[cols])[:,1]
    pred=(score>=0.5).astype(int)
    m=make_metrics(yte,pred,score)
    m["FeatureGroup"]=group
    rows.append(m)

model=ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
model.fit(Xtr[OFFICIAL_FEATURES], ytr)
score=model.predict_proba(Xte[OFFICIAL_FEATURES])[:,1]
pred=(score>=0.5).astype(int)
m=make_metrics(yte,pred,score); m["FeatureGroup"]="All 18 features"; rows.append(m)

out=pd.DataFrame(rows)[["FeatureGroup","Accuracy","Precision","Recall","F1","MCC","ROC-AUC","PR-AUC"]]
out.to_csv(Path(__file__).resolve().parents[1]/"results/feature_ablation_reproduced.csv", index=False)
print(out)
