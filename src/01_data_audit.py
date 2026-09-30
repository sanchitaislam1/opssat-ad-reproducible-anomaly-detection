import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from common import load_segments, split_xy, OFFICIAL_FEATURES

df = load_segments()
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nTrain/test:")
print(df["train"].value_counts().sort_index())
print("\nLabels:")
print(df["anomaly"].value_counts().sort_index())
print("\nChannels:", df["channel"].nunique())
print("Missing cells:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

tr = df["train"].eq(1)
print("\nTraining class counts:")
print(df.loc[tr, "anomaly"].value_counts())
print("\nTest class counts:")
print(df.loc[~tr, "anomaly"].value_counts())

feature_rows = df[OFFICIAL_FEATURES].copy()
print("\nExact duplicate feature rows:", int(feature_rows.duplicated().sum()))
print("Identifier/metadata excluded: segment, channel, sampling")
