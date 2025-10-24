import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np
from datetime import datetime, timedelta, time

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("xgb_waterlevel_hourly_model.pkl")

st.title("🌊 Water Level Forecast Dashboard")

# -----------------------------
# Current time (GMT+7), rounded up to next full hour
# -----------------------------
now_utc = datetime.utcnow()
gmt7_now = now_utc + timedelta(hours=7)
rounded_now = (gmt7_now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0) \
    if gmt7_now.minute > 0 or gmt7_now.second > 0 else gmt7_now.replace(minute=0, second=0, microsecond=0)

# -----------------------------
# Select forecast start datetime
# -----------------------------
st.subheader("Select Start Date & Time for 7-Day Forecast")
selected_date = st.date_input("Date", value=rounded_now.date(), max_value=rounded_now.date())
hour_options = [f"{h:02d}:00" for h in range(0, rounded_now.hour + 1)]
selected_hour_str = st.selectbox("Hour", hour_options, index=len(hour_options)-1)
selected_hour = int(selected_hour_str.split(":")[0])
start_datetime = datetime.combine(selected_date, time(selected_hour, 0, 0))
st.write(f"Start datetime (GMT+7): {start_datetime}")

# -----------------------------
# Upload water level data
# -----------------------------
st.subheader("Upload Hourly Water Level File")
uploaded_file = st.file_uploader("Upload CSV File (AWLR Joloi Logs)", type=["csv"])
wl_hourly = None

if uploaded_file is not None:
    df_wl = pd.read_csv(uploaded_file, engine='python', skip_blank_lines=True)
    if "Datetime" not in df_wl.columns or "Level Air" not in df_wl.columns:
        st.error("The file must contain columns 'Datetime' and 'Level Air'.")
    else:
        df_wl["Datetime"] = pd.to_datetime(df_wl["Datetime"]).dt.floor("H")
        start_limit = start_datetime - pd.Timedelta(hours=24)
        wl_hourly = (df_wl[(df_wl["Datetime"] >= start_limit) & (df_wl["Datetime"] < start_datetime)]
                     .groupby("Datetime")["Level Air"].mean()
                     .reset_index()
                     .rename(columns={"Level Air": "Water_level"})
                     .sort_values("Datetime")
                     .round(2))
        st.success("Successfully uploaded 24-hour water level data before start time.")
        st.dataframe(wl_hourly)

# -----------------------------
# Fetch climate data
# -----------------------------
def fetch_climate_historical(start_dt, end_dt, lat=-0.117, lon=114.1):
    start_date = start_dt.date().isoformat()
    end_date = end_dt.date().isoformat()
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=surface_pressure,cloud_cover,soil_temperature_0_to_7cm,soil_moisture_0_to_7cm,rain&timezone=Asia%2FBangkok"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame({
            "Datetime": pd.to_datetime(data["hourly"]["time"]),
            "Rainfall": data["hourly"]["rain"],
            "Cloud_cover": data["hourly"]["cloud_cover"],
            "Surface_pressure": data["hourly"]["surface_pressure"],
            "Soil_temperature": data["hourly"]["soil_temperature_0_to_7cm"],
            "Soil_moisture": data["hourly"]["soil_moisture_0_to_7cm"]
        })
        df["Datetime"] = df["Datetime"].dt.floor("H")
        return df
    except:
        return pd.DataFrame()

def fetch_climate_forecast(lat=-0.117, lon=114.1):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=rain,surface_pressure,cloud_cover,soil_moisture_0_to_1cm,soil_temperature_0cm&timezone=Asia%2FBangkok&forecast_days=14"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame({
            "Datetime": pd.to_datetime(data["hourly"]["time"]),
            "Rainfall": data["hourly"]["rain"],
            "Cloud_cover": data["hourly"]["cloud_cover"],
            "Surface_pressure": data["hourly"]["surface_pressure"],
            "Soil_temperature": data["hourly"]["soil_temperature_0cm"],
            "Soil_moisture": data["hourly"]["soil_moisture_0_to_1cm"]
        })
        df["Datetime"] = df["Datetime"].dt.floor("H")
        return df
    except:
        return pd.DataFrame()

# -----------------------------
# Lag features helper
# -----------------------------
def create_lag_features(df, lag_config):
    df_lag = df.copy()
    for col, lags in lag_config.items():
        for lag in lags:
            df_lag[f"{col}_Lag{lag}"] = df_lag[col].shift(lag)
    return df_lag

# -----------------------------
# Run forecast with progress bar
# -----------------------------
if wl_hourly is not None:
    if st.button("Run 7-Day Forecast"):
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- historical + last 24h climate ---
        status_text.text("Fetching historical climate data...")
        hist_climate = fetch_climate_historical(start_datetime - timedelta(hours=24), start_datetime)
        df_hist = pd.merge(wl_hourly, hist_climate, on="Datetime", how="left").sort_values("Datetime")
        
        # --- forecast period ---
        status_text.text("Fetching forecast climate data...")
        forecast_hours = 168
        forecast_index = [start_datetime + timedelta(hours=i) for i in range(forecast_hours)]
        df_forecast = pd.DataFrame({"Datetime": forecast_index})
        forecast_climate = fetch_climate_forecast()
        df_forecast = pd.merge(df_forecast, forecast_climate, on="Datetime", how="left")
        df_forecast["Water_level"] = np.nan
        
        # --- concat full ---
        df_full = pd.concat([df_hist, df_forecast], ignore_index=True).sort_values("Datetime").reset_index(drop=True)
        
        # --- lag configuration ---
        lag_config = {
            "Rainfall": list(range(17,25)),
            "Cloud_cover": list(range(1,25)),
            "Surface_pressure": list(range(1,25)),
            "Soil_temperature": list(range(9,12)),
            "Soil_moisture": list(range(1,25)),
            "Water_level": list(range(1,25))
        }
        
        # --- iterative prediction ---
        status_text.text("Running iterative prediction...")
        for i in range(len(df_hist), len(df_full)):
            df_lagged = create_lag_features(df_full.iloc[:i], lag_config)
            lag_cols = [c for c in df_lagged.columns if "_Lag" in c]
            X_pred = df_lagged[lag_cols].iloc[-1].values.reshape(1, -1)
            df_full.at[i, "Water_level"] = model.predict(X_pred)[0]
            
            # update progress bar
            progress = int((i - len(df_hist) + 1) / forecast_hours * 100)
            progress_bar.progress(progress)
        
        # --- final display ---
        display_cols = ["Datetime","Water_level","Rainfall","Cloud_cover","Surface_pressure","Soil_temperature","Soil_moisture"]
        final_display = df_full[display_cols].copy()
        final_display["Source"] = ["Historical"]*len(df_hist) + ["Forecast"]*len(df_forecast)
        
        # drop NaN rows
        numeric_cols = final_display.select_dtypes(include=np.number).columns
        final_display = final_display.dropna(subset=numeric_cols)
        
        # styling & 2 decimal places
        styled_df = (final_display.style
                     .apply(lambda row: ['background-color: #cfe9ff' if row.Source=='Forecast' else '' for _ in row], axis=1)
                     .format("{:.2f}", subset=numeric_cols))
        
        status_text.text("Forecast completed!")
        st.subheader("Water Level + Climate Forecast")
        st.dataframe(styled_df, use_container_width=True, height=500)
