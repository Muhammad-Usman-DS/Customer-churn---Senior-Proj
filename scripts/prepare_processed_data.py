import os, sys
import pandas as pd

# make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data 
from src.features.build_features import build_features 

RAW = "/Users/usmangoat/Desktop/Portfolio Prorjects/Customer churn - Senior Proj/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT = "data/processed/telco_churn_processed.csv"

# 1) load raw
df = pd.read_csv(RAW)

# 2) preprocess (drops id, fixes TotalCharges, etc.)
df = preprocess_data(df, target_col="Churn")

# 3) ensure target is 0/1 only if still object
if "Churn" in df.columns and df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1}).astype("Int64")

# sanity checks
assert df["Churn"].isna().sum() == 0, "Churn has NaNs after preprocess"
assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1 after preprocess"

# 4) features
df_processed = build_features(df, target_col="Churn")

# 5) save
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_processed.to_csv(OUT, index=False)
print(f" - Processed dataset saved to {OUT} | Shape: {df_processed.shape}")

