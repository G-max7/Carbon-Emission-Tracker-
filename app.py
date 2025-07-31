from flask import Flask, jsonify, render_template
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import threading
import time
from datetime import datetime
from src.predict import predict_emissions
import requests
from twilio.rest import Client
from dotenv import load_dotenv
from collections import deque

# Load environment variables
load_dotenv()

# Initialize Twilio client
twilio_client = Client(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN")
)

# Debug Twilio credentials
TWILIO_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
INCHARGE_NUMBER = "+918248179868"
print("Twilio FROM number:", TWILIO_NUMBER)
print("Twilio TO number:", INCHARGE_NUMBER)

app = Flask(__name__)

# Constants
RAW_DATA_PATH = "data/raw/sensor_data.csv"
CARBON_THRESHOLD = 40
SOS_THRESHOLD = 45
MAX_SOS_COUNT = 1

# Ensure raw data folder exists
os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

expected_features = [
    "PM2.5 (ug/m3)", "PM10 (ug/m3)", "NO (ug/m3)", "NO2 (ug/m3)", "NOx (ppb)",
    "NH3 (ug/m3)", "SO2 (ug/m3)", "CO (mg/m3)", "Ozone (ug/m3)", "Benzene (ug/m3)",
    "Toluene (ug/m3)", "Temp (degree C)", "RH (%)", "WS (m/s)", "WD (deg)",
    "SR (W/mt2)", "BP (mmHg)", "VWS (m/s)", "Xylene (ug/m3)", "RF (mm)", "AT (degree C)"
]

sos_counter = 0

def get_tyres_suggestion(emission_value):
    prompt = (
        f"The predicted carbon emission is {emission_value:.2f}, which exceeds the threshold of {CARBON_THRESHOLD}. "
        "Suggest a practical strategy for reducing emissions specifically in the tyre manufacturing industry."
    )

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
            json={"inputs": prompt},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list):
                return result[0]["generated_text"].split(prompt)[-1].strip()
            return result.get("generated_text", "No suggestion found.")
        else:
            print("🛑 Hugging Face API error:", response.status_code, response.text)
            return "No suggestion available at the moment."

    except Exception as e:
        print(f"❌ Suggestion fetch error: {e}")
        return "No suggestion available due to an error."

def generate_sensor_data():
    current_hour = datetime.now().hour
    base_values = {
        "PM2.5 (ug/m3)": (30, 10),
        "PM10 (ug/m3)": (70, 20),
        "NO (ug/m3)": (20, 5),
        "NO2 (ug/m3)": (15, 4),
        "NOx (ppb)": (25, 8),
        "NH3 (ug/m3)": (5, 1.5),
        "SO2 (ug/m3)": (30, 6),
        "CO (mg/m3)": (2.5, 0.5),
        "Ozone (ug/m3)": (25, 5),
        "Benzene (ug/m3)": (0.2, 0.05),
        "Toluene (ug/m3)": (0.3, 0.07),
        "Temp (degree C)": (32, 2),
        "RH (%)": (60, 5),
        "WS (m/s)": (3, 0.5),
        "WD (deg)": (270, 10),
        "SR (W/mt2)": (150, 20),
        "BP (mmHg)": (1015, 10),
        "VWS (m/s)": (2, 0.5),
        "Xylene (ug/m3)": (0.4, 0.1),
        "RF (mm)": (1, 0.2),
        "AT (degree C)": (33, 2),
    }

    if 6 <= current_hour < 10:
        multiplier = 1.3
    elif 10 <= current_hour < 16:
        multiplier = 0.8
    elif 16 <= current_hour < 20:
        multiplier = 1.5
    else:
        multiplier = 0.7

    sensor_data = {
        key: max(0, np.random.normal(mu * multiplier, sigma))
        for key, (mu, sigma) in base_values.items()
    }
    sensor_data["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sensor_data["From Date"] = datetime.now().strftime("%Y-%m-%d")
    return sensor_data

def save_sensor_data():
    print("📡 Starting sensor data simulation...")
    while True:
        try:
            data = generate_sensor_data()
            missing_keys = [f for f in expected_features if f not in data]
            if missing_keys:
                print(f"⚠️ Filling missing features with default 0: {missing_keys}")
                for key in missing_keys:
                    data[key] = 0

            df = pd.DataFrame([data])
            df.to_csv(RAW_DATA_PATH, mode="a", header=not os.path.exists(RAW_DATA_PATH), index=False)

            try:
                predicted_emission = predict_emissions(data)
                print(f"🌿 Predicted Emission: {predicted_emission:.2f}")

                global sos_counter
                if predicted_emission >= SOS_THRESHOLD:
                    sos_counter += 1
                    print(f"⚠️ Emission above threshold. Consecutive count: {sos_counter}")
                    if sos_counter >= 5:
                        print("🚨 Emission exceeded threshold 5 times. Sending SOS SMS...")
                        if TWILIO_NUMBER and INCHARGE_NUMBER:
                            alert_msg = (
                                f"🚨 SOS ALERT: Emissions exceeded {SOS_THRESHOLD} ppm 5 times in a row. "
                                f"Current emission: {predicted_emission:.2f} ppm."
                            )
                            try:
                                msg = twilio_client.messages.create(
                                    body=alert_msg,
                                    from_=TWILIO_NUMBER,
                                    to=INCHARGE_NUMBER
                                )
                                print("✅ SOS alert sent. SID:", msg.sid)
                                if msg.sid:
                                    print("📨 SMS successfully delivered.")
                                else:
                                    print("⚠️ SMS delivery status unknown.")
                            except Exception as sms_error:
                                print("❌ Twilio SMS send failed:", sms_error)
                                print("📨 SMS failed to send.")
                        sos_counter = 0
                else:
                    sos_counter = 0

            except Exception as pe:
                print(f"❌ Prediction error: {pe}")

            time.sleep(5)

        except Exception as e:
            print(f"❌ Sensor save error: {e}")

@app.route("/")
def dashboard():
    return render_template("index.html")

@app.route("/live-data", methods=["GET"])
def get_live_data():
    try:
        if not os.path.exists(RAW_DATA_PATH):
            return jsonify({"status": "error", "message": "No data available."}), 404

        df = pd.read_csv(RAW_DATA_PATH, on_bad_lines='skip').dropna()
        if df.empty:
            return jsonify({"status": "error", "message": "No valid data."}), 404

        latest = df.iloc[-1].to_dict()

        missing = [f for f in expected_features if f not in latest]
        if missing:
            raise ValueError(f"❌ Missing features: {missing}")

        predicted_carbon = predict_emissions(latest)
        suggestion = get_tyres_suggestion(predicted_carbon) if predicted_carbon > CARBON_THRESHOLD else None

        response = {
            "status": "success",
            "latest_sensor_data": latest,
            "predicted_carbon": round(predicted_carbon, 2),
            "threshold": CARBON_THRESHOLD,
            "suggestion": suggestion,
        }

        return jsonify(response)

    except Exception as e:
        print(f"🔥 Error in /live-data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/trigger-sos", methods=["POST"])
def trigger_sos():
    try:
        alert_msg = "🚨 SOS ALERT: Emission exceeded the safe limit 5 times in a row!"
        msg = twilio_client.messages.create(
            body=alert_msg,
            from_=TWILIO_NUMBER,
            to=INCHARGE_NUMBER
        )
        print("✅ SOS SMS triggered manually. SID:", msg.sid)
        return jsonify({"status": "success", "sid": msg.sid})
    except Exception as e:
        print(f"❌ Error sending manual SOS SMS: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    threading.Thread(target=save_sensor_data, daemon=True).start()
    app.run(host="0.0.0.0", port=5002, debug=True)