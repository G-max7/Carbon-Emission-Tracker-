from flask import Flask, jsonify, render_template
from threading import Thread
import time
import random
from datetime import datetime
import sys
import os

# Add parent path for importing predict_emissions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_emissions

# Explicit path to dashboard/templates
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'dashboard', 'templates')

app = Flask(__name__, template_folder=TEMPLATES_DIR)

latest_data = {}

def generate_fake_sensor_data():
    """Mock function to simulate real-time sensor data updates."""
    while True:
        global latest_data
        latest_data = {
            "PM2.5 (ug/m3)": random.uniform(10, 100),
            "PM10 (ug/m3)": random.uniform(20, 150),
            "NO (ug/m3)": random.uniform(5, 50),
            "NO2 (ug/m3)": random.uniform(5, 50),
            "NOx (ppb)": random.uniform(5, 50),
            "NH3 (ug/m3)": random.uniform(0, 10),
            "SO2 (ug/m3)": random.uniform(5, 30),
            "CO (mg/m3)": random.uniform(0.5, 5),
            "Ozone (ug/m3)": random.uniform(10, 100),
            "Benzene (ug/m3)": random.uniform(0, 1),
            "Toluene (ug/m3)": random.uniform(0, 1),
            "Temp (degree C)": random.uniform(20, 40),
            "RH (%)": random.uniform(30, 90),
            "WS (m/s)": random.uniform(0, 5),
            "WD (deg)": random.uniform(0, 360),
            "SR (W/mt2)": random.uniform(100, 1000),
            "BP (mmHg)": random.uniform(750, 800),
            "VWS (m/s)": random.uniform(0, 5),
            "Xylene (ug/m3)": random.uniform(0, 1),
            "RF (mm)": random.uniform(0, 1),
            "AT (degree C)": random.uniform(20, 40),
            "From Date": datetime.now().strftime('%Y-%m-%d'),
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        try:
            # Get emission prediction
            prediction = predict_emissions(latest_data)
            latest_data["Predicted Emissions"] = round(prediction, 2)
        except Exception as e:
            print("⚠️ Prediction failed:", e)
            latest_data["Predicted Emissions"] = "N/A"

        print("✅ Updated data:", latest_data)
        time.sleep(5)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/live-data")
def live_data():
    return jsonify(latest_data)

if __name__ == "__main__":
    thread = Thread(target=generate_fake_sensor_data)
    thread.daemon = True
    thread.start()
    app.run(debug=True)
