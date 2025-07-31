import joblib
import pandas as pd
import os

# ✅ Load the trained model
model_path = "models/emissions_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}. Train the model first.")

model = joblib.load(model_path)

# ✅ Define expected features (including 'From Date' used during training)
EXPECTED_FEATURES = [
    "From Date",  # Keep this as-is to match training
    "PM2.5 (ug/m3)", "PM10 (ug/m3)", "NO (ug/m3)", "NO2 (ug/m3)", "NOx (ppb)", 
    "NH3 (ug/m3)", "SO2 (ug/m3)", "CO (mg/m3)", "Ozone (ug/m3)", "Benzene (ug/m3)", 
    "Toluene (ug/m3)", "Temp (degree C)", "RH (%)", "WS (m/s)", "WD (deg)", 
    "SR (W/mt2)", "BP (mmHg)", "VWS (m/s)", "Xylene (ug/m3)", "RF (mm)", "AT (degree C)"
]

def predict_emissions(data):
    """
    Predict emissions based on sensor input.

    Args:
        data (dict): Dictionary containing sensor values.

    Returns:
        float: Predicted emissions value.
    """

    # ✅ Ensure "From Date" exists
    data["From Date"] = "2025-04-03"  # Placeholder date to match training

    # ✅ Ensure all required features exist
    missing_features = [feature for feature in EXPECTED_FEATURES if feature not in data]
    if missing_features:
        raise ValueError(f"❌ Missing features: {missing_features}")

    # ✅ Convert input to DataFrame
    df = pd.DataFrame([data])

    # ✅ Ensure correct column order
    df = df[EXPECTED_FEATURES]

    # ✅ Convert to numeric where possible
    df = df.apply(pd.to_numeric, errors='coerce')

    # ✅ Fill missing values correctly
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)

    # ✅ Make prediction
    try:
        prediction = model.predict(df)[0]
        return prediction
    except Exception as e:
        raise RuntimeError(f"❌ Prediction failed: {str(e)}")


# ✅ Test if script runs directly
if __name__ == "__main__":
    test_data = {
        "PM2.5 (ug/m3)": 30, "PM10 (ug/m3)": 50, "NO (ug/m3)": 20, "NO2 (ug/m3)": 15, "NOx (ppb)": 10,
        "NH3 (ug/m3)": 5, "SO2 (ug/m3)": 8, "CO (mg/m3)": 0.9, "Ozone (ug/m3)": 25, "Benzene (ug/m3)": 0.2,
        "Toluene (ug/m3)": 0.3, "Temp (degree C)": 32, "RH (%)": 60, "WS (m/s)": 3, "WD (deg)": 270,
        "SR (W/mt2)": 150, "BP (mmHg)": 1015, "VWS (m/s)": 2, "Xylene (ug/m3)": 0.4, "RF (mm)": 1,
        "AT (degree C)": 33
    }

    try:
        prediction = predict_emissions(test_data)
        print(f"✅ Predicted Emissions: {prediction}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
