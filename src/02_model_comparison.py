import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from common import load_segments, split_xy, make_metrics
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

df = load_segments()
Xtr, Xte, ytr, yte = split_xy(df)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
    "SVM-RBF": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42, max_iter=200),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=-1
    ),
}

rows=[]
for name, model in models.items():
    model.fit(Xtr, ytr)
    score = model.predict_proba(Xte)[:,1] if hasattr(model, "predict_proba") else model.decision_function(Xte)
    pred = (score >= 0.5).astype(int)
    m = make_metrics(yte, pred, score)
    m["Model"]=name
    rows.append(m)
    print(name, m)

out=pd.DataFrame(rows)[["Model","Accuracy","Precision","Recall","F1","MCC","ROC-AUC","PR-AUC"]]
out.to_csv(Path(__file__).resolve().parents[1]/"results/model_comparison_reproduced.csv", index=False)
print(out)
