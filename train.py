import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# 📌 Load preprocessed dataset
data_path = "data/processed/emissions_data.csv"

try:
    df = pd.read_csv(data_path)
    print(f"✅ Successfully loaded dataset from: {data_path}")
except FileNotFoundError:
    print(f"❌ ERROR: File not found at {data_path}. Ensure preprocessing is complete.")
    exit()

# ✅ Trim column names (remove extra spaces)
df.columns = df.columns.str.strip()

# ✅ Display dataset info
print(f"📊 Dataset Shape: {df.shape}")
print(f"🔍 Columns in dataset: {df.columns.tolist()}")

# ✅ Ensure numeric data types
df = df.apply(pd.to_numeric, errors='coerce')

# ✅ Handle missing values (forward & backward fill)
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)

# ✅ Ensure 'target' column exists
if "target" not in df.columns:
    print("⚠️ WARNING: 'target' column not found! Attempting to auto-generate...")
    # Auto-generate target as sum of major gas emissions
    gas_columns = ["NO2 (ug/m3)", "CO (mg/m3)", "SO2 (ug/m3)", "PM10 (ug/m3)"]
    missing_gases = [col for col in gas_columns if col not in df.columns]

    if missing_gases:
        print(f"❌ ERROR: Cannot generate 'target'. Missing columns: {missing_gases}")
        exit()
    
    df["target"] = df[gas_columns].sum(axis=1)
    print("✅ 'target' column generated successfully!")

# 📌 Define Features (X) and Target (y)
X = df.drop(columns=["target"])
y = df["target"]

# 📌 Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 Evaluate model performance
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model Training Complete! R² Score: {r2:.4f}")

# 📌 Save the trained model
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "emissions_model.pkl")
joblib.dump(model, model_path)

print(f"✅ Model saved at: {model_path}")
