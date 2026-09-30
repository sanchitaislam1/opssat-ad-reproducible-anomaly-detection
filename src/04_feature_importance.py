import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from common import load_segments, split_xy, OFFICIAL_FEATURES
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
import pandas as pd

df=load_segments()
Xtr,Xte,ytr,yte=split_xy(df)
model=ExtraTreesClassifier(n_estimators=500,class_weight="balanced",random_state=42,n_jobs=-1)
model.fit(Xtr,ytr)

r=permutation_importance(model,Xte,yte,scoring="f1",n_repeats=20,random_state=42,n_jobs=-1)
out=pd.DataFrame({"Feature":OFFICIAL_FEATURES,"Importance":r.importances_mean})
out=out.sort_values("Importance",ascending=False)
out.to_csv(Path(__file__).resolve().parents[1]/"results/permutation_importance.csv",index=False)
print(out.head(18))
