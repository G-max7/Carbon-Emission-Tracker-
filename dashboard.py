from flask import Flask, render_template_string, jsonify
import pandas as pd
import os
import plotly.graph_objs as go
from datetime import datetime
from src.predict import predict_emissions

app = Flask(__name__)

GOV_LIMITS = {
    "PM2.5 (ug/m3)": 60,
    "PM10 (ug/m3)": 100,
    "NOx (ppb)": 40,
    "SO2 (ug/m3)": 50,
    "CO (mg/m3)": 4
}

warning_counter = {gas: 0 for gas in GOV_LIMITS}

suggestions = {
    "PM2.5 (ug/m3)": "Improve dust collection filters",
    "PM10 (ug/m3)": "Use water sprays to reduce airborne particles",
    "NOx (ppb)": "Optimize combustion temperatures",
    "SO2 (ug/m3)": "Switch to low-sulfur fuels",
    "CO (mg/m3)": "Improve ventilation and fuel efficiency"
}

def load_sensor_data():
    try:
        sensor_data = pd.read_csv("data/raw/sensor_data.csv")
        return sensor_data.tail(10)
    except Exception as e:
        return str(e)

def generate_full_input(row):
    default_values = {
        "NO (ug/m3)": 20, "NO2 (ug/m3)": 15, "NH3 (ug/m3)": 5,
        "Ozone (ug/m3)": 25, "Benzene (ug/m3)": 0.2, "Toluene (ug/m3)": 0.3,
        "Temp (degree C)": 32, "RH (%)": 60, "WS (m/s)": 3, "WD (deg)": 270,
        "SR (W/mt2)": 150, "BP (mmHg)": 1015, "VWS (m/s)": 2, "Xylene (ug/m3)": 0.4,
        "RF (mm)": 1, "AT (degree C)": 33
    }
    row = row.copy()
    for k, v in default_values.items():
        if k not in row or pd.isnull(row[k]):
            row[k] = v
    if "From Date" not in row or pd.isnull(row["From Date"]):
        row["From Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return row

def predict_from_sensor():
    try:
        df = pd.read_csv("data/raw/sensor_data.csv")
        latest_row = df.iloc[-1]
        input_data = generate_full_input(latest_row)
        emission_value = predict_emissions(input_data)

        exceeded = {}
        suggestions_triggered = []

        for gas in GOV_LIMITS:
            value = latest_row.get(gas)
            if pd.notnull(value) and value > GOV_LIMITS[gas]:
                warning_counter[gas] += 1
                exceeded[gas] = value

                if warning_counter[gas] >= 5:
                    suggestions_triggered.append((gas, suggestions[gas]))
            else:
                warning_counter[gas] = 0

        return {
            "predicted_emission": round(emission_value, 2),
            "exceeded": exceeded,
            "suggestions": suggestions_triggered
        }
    except Exception as e:
        return str(e)

def generate_bar_chart(values_dict):
    if not values_dict:
        return "<p>No gases exceeded the limits.</p>"
    bars = go.Bar(x=list(values_dict.keys()), y=list(values_dict.values()), marker_color='indianred')
    layout = go.Layout(
        title='Exceeded Gases vs Limits',
        xaxis=dict(title='Gas'),
        yaxis=dict(title='Level'),
        template='plotly_white'
    )
    fig = go.Figure(data=[bars], layout=layout)
    return fig.to_html(full_html=False)

def generate_trend_chart():
    try:
        df = pd.read_csv("data/raw/sensor_data.csv")
        recent = df.tail(20)
        emissions = []
        timestamps = []

        for _, row in recent.iterrows():
            input_data = generate_full_input(row)
            pred = predict_emissions(input_data)
            emissions.append(round(pred, 2))
            timestamps.append(row.get("From Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        line = go.Scatter(x=timestamps, y=emissions, mode='lines+markers', name='Emission Trend')
        layout = go.Layout(title='Predicted Emission Trend', xaxis_title='Time', yaxis_title='Emission Value')
        fig = go.Figure(data=[line], layout=layout)
        return fig.to_html(full_html=False)
    except Exception as e:
        return f"<p>Trend chart error: {e}</p>"

@app.route('/')
def index():
    sensor_data = load_sensor_data()
    prediction = predict_from_sensor()
    return jsonify(
        sensor_data=sensor_data.to_dict(orient='records') if isinstance(sensor_data, pd.DataFrame) else sensor_data,
        prediction=prediction
    )

if __name__ == '__main__':
    app.run(debug=True, port=5002)
