from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "segments.csv"

OFFICIAL_FEATURES = [
    "mean", "var", "std", "kurtosis", "skew",
    "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks",
    "diff_peaks", "diff2_peaks", "diff_var", "diff2_var",
    "duration", "len", "len_weighted", "gaps_squared",
    "var_div_duration", "var_div_len",
]

FEATURE_GROUPS = {
    "Statistical": ["mean","var","std","kurtosis","skew"],
    "Peak/shape": ["n_peaks","smooth10_n_peaks","smooth20_n_peaks"],
    "Derivative": ["diff_peaks","diff2_peaks","diff_var","diff2_var"],
    "Length/gap": ["duration","len","len_weighted","gaps_squared",
                   "var_div_duration","var_div_len"],
}

def load_segments(path=DATA):
    return pd.read_csv(path)

def split_xy(df):
    train = df["train"].astype(int).eq(1)
    Xtr = df.loc[train, OFFICIAL_FEATURES].copy()
    ytr = df.loc[train, "anomaly"].astype(int).copy()
    Xte = df.loc[~train, OFFICIAL_FEATURES].copy()
    yte = df.loc[~train, "anomaly"].astype(int).copy()
    return Xtr, Xte, ytr, yte

def make_metrics(y_true, pred, score):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        matthews_corrcoef, roc_auc_score, average_precision_score
    )
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
        "ROC-AUC": roc_auc_score(y_true, score),
        "PR-AUC": average_precision_score(y_true, score),
    }
