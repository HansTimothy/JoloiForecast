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
if gmt7_now.minute > 0 or gmt7_now.second > 0 or gmt7_now.microsecond > 0:
    rounded_now = (gmt7_now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
else:
    rounded_now = gmt7_now.replace(minute=0, second=0, microsecond=0)

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
    try:
        df_wl = pd.read_csv(uploaded_file, engine='python', skip_blank_lines=True)
        if "Datetime" not in df_wl.columns or "Level Air" not in df_wl.columns:
            st.error("The file must contain columns 'Datetime' and 'Level Air'.")
        else:
            df_wl["Datetime"] = pd.to_datetime(df_wl["Datetime"]).dt.floor("H")
            start_limit = start_datetime - pd.Timedelta(hours=24)
            df_wl_filtered = df_wl[(df_wl["Datetime"] >= start_limit) & (df_wl["Datetime"] < start_datetime)]
            wl_hourly = (
                df_wl_filtered.groupby("Datetime")["Level Air"].mean().reset_index()
                .rename(columns={"Level Air": "Water_level"})
                .sort_values(by="Datetime")
                .round(2)
            )
            st.success("Successfully uploaded 24-hour water level data before start time.")
            st.dataframe(wl_hourly)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

# -----------------------------
# Fetch climate data
# -----------------------------
def fetch_climate_historical(start_dt, end_dt, lat=-0.117, lon=114.1):
    start_date = start_dt.date().isoformat()
    end_date = end_dt.date().isoformat()
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=surface_pressure,cloud_cover,soil_temperature_0_to_7cm,soil_moisture_0_to_7cm,rain"
        f"&timezone=Asia%2FBangkok"
    )
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
    except Exception as e:
        st.error(f"Failed to fetch historical climate data: {e}")
        return pd.DataFrame()

def fetch_climate_forecast(lat=-0.117, lon=114.1):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=rain,surface_pressure,cloud_cover,soil_moisture_0_to_1cm,soil_temperature_0cm"
        f"&timezone=Asia%2FBangkok&forecast_days=14"
    )
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
    except Exception as e:
        st.error(f"Failed to fetch forecast climate data: {e}")
        return pd.DataFrame()

# -----------------------------
# Lag feature helper
# -----------------------------
def create_lag_features(df, lag_config):
    df_lag = df.copy()
    for col, lags in lag_config.items():
        for lag in lags:
            df_lag[f"{col}_Lag{lag}"] = df_lag[col].shift(lag)
    return df_lag

# -----------------------------
# Forecast iterative
# -----------------------------
if wl_hourly is not None:
    if st.button("Run 7-Day Forecast"):
        with st.spinner("Fetching climate data and performing 7-day forecast"):
            # Fetch climate for forecast period
            forecast_hours = 168  # 7x24
            climate_forecast = fetch_climate_forecast()
            
            # Merge last historical water level + climate
            last_hist = wl_hourly.copy()
            last_climate = fetch_climate_historical(start_datetime - timedelta(hours=24), start_datetime)
            df_hist = pd.merge(last_hist, last_climate, on="Datetime", how="left").sort_values("Datetime")
            
            # Prepare forecast DataFrame
            forecast_index = [start_datetime + timedelta(hours=i) for i in range(forecast_hours)]
            df_forecast = pd.DataFrame({"Datetime": forecast_index})
            df_forecast["Water_level"] = np.nan
            df_forecast = pd.merge(df_forecast, climate_forecast, on="Datetime", how="left")
            
            # Lag configuration: hanya untuk highlighted features
            lag_config = {
                "Rainfall": list(range(17,25)),
                "Cloud_cover": list(range(1,25)),
                "Surface_pressure": list(range(1,25)),
                "Soil_temperature": list(range(9,12)),
                "Soil_moisture": list(range(1,25)),
                "Water_level": list(range(1,25))
            }
            
            # Concatenate historical + forecast for iterative pred
            df_full = pd.concat([df_hist, df_forecast], ignore_index=True).sort_values("Datetime").reset_index(drop=True)
            
            # Iterative prediction
            for i in range(len(df_hist), len(df_full)):
                df_lagged = create_lag_features(df_full.iloc[:i], lag_config)
                lag_cols = [c for c in df_lagged.columns if "_Lag" in c]
                X_pred = df_lagged[lag_cols].iloc[-1].values.reshape(1, -1)
                y_hat = model.predict(X_pred)[0]
                df_full.at[i, "Water_level"] = y_hat
            
            # -----------------------------
            # Prepare final display
            # -----------------------------
            final_display = df_full[["Datetime","Water_level","Rainfall","Cloud_cover",
                                     "Surface_pressure","Soil_temperature","Soil_moisture"]].copy()
            final_display["Source"] = ["Historical"]*len(df_hist) + ["Forecast"]*len(df_forecast)
            
            # Round numeric columns to 2 decimals
            numeric_cols = final_display.select_dtypes(include=np.number).columns
            final_display[numeric_cols] = final_display[numeric_cols].round(2)
            
            # Drop rows with any NaN
            final_display = final_display.dropna(subset=numeric_cols)
            
            # Highlight forecast rows
            def highlight_forecast(row):
                color = 'background-color: #cfe9ff' if row['Source']=="Forecast" else ''
                return [color]*len(row)
            
            styled_df = final_display.style.apply(highlight_forecast, axis=1)
            st.subheader("Water Level + Climate Forecast")
            st.dataframe(styled_df, use_container_width=True, height=500)
