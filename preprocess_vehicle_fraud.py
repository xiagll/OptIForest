import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Load raw dataset
# --------------------------------------------------

df = pd.read_csv("data/fraud_oracle.csv")

print("Original shape:", df.shape)

# --------------------------------------------------
# 2. Remove identifier column
# --------------------------------------------------

df = df.drop(columns=["PolicyNumber"])

# --------------------------------------------------
# 3. Handle invalid / unknown values
# --------------------------------------------------

# One invalid '0' value exists in each of these columns.
# Replace with the most common value (mode).
for col in ["DayOfWeekClaimed", "MonthClaimed"]:
    df[col] = df[col].replace("0", np.nan)
    df[col] = df[col].fillna(df[col].mode()[0])

# Age = 0 is treated as unknown.
df["Age"] = df["Age"].replace(0, np.nan)
df["Age"] = df["Age"].fillna(df["Age"].median())

# --------------------------------------------------
# 4. Separate target label
# --------------------------------------------------

y = df["FraudFound_P"].astype(int)

X = df.drop(columns=["FraudFound_P"])

# --------------------------------------------------
# 5. Treat identifier-like numeric feature as categorical
# --------------------------------------------------

X["RepNumber"] = X["RepNumber"].astype(str)

# Year only has three categories, so treat it as categorical.
X["Year"] = X["Year"].astype(str)

# --------------------------------------------------
# 6. Identify categorical and numerical columns
# --------------------------------------------------

categorical_columns = X.select_dtypes(include=["object", "str"]).columns.tolist()
numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

print("Categorical columns:", len(categorical_columns))
print("Numeric columns:", len(numeric_columns))

# --------------------------------------------------
# 7. One-hot encode categorical variables
# --------------------------------------------------

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=False,
    dtype=int
)

# --------------------------------------------------
# 8. Scale original numerical variables
# --------------------------------------------------

scaler = StandardScaler()

X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# --------------------------------------------------
# 9. Put anomaly label back as FINAL column
# --------------------------------------------------

processed = X.copy()
processed["FraudFound_P"] = y.values

# --------------------------------------------------
# 10. Save in format expected by OptIForest
# --------------------------------------------------

processed.to_csv(
    "data/vehicle_fraud.csv",
    index=False,
    header=False
)

print("\nProcessed shape:", processed.shape)
print("Missing values:", processed.isnull().sum().sum())

print("\nClass distribution:")
print(processed["FraudFound_P"].value_counts())

print("\nSaved to: data/vehicle_fraud.csv")