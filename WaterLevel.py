import streamlit as st
import pandas as pd
import requests
import joblib
import plotly.graph_objects as go
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

selected_date = st.date_input(
    "Date",
    value=rounded_now.date(),
    max_value=rounded_now.date()
)

hour_options = [f"{h:02d}:00" for h in range(0, rounded_now.hour + 1)]
selected_hour_str = st.selectbox(
    "Hour",
    hour_options,
    index=len(hour_options)-1
)

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
                .sort_values(by="Datetime", ascending=True)
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
# Merge data & forecasting
# -----------------------------
if wl_hourly is not None:
    if st.button("Fetch Climate Data & Forecasting"):
        # Historical climate
        start_dt = wl_hourly["Datetime"].min()
        end_dt = wl_hourly["Datetime"].max()
        climate_hist = fetch_climate_historical(start_dt, end_dt)
        merged_df = pd.merge(wl_hourly, climate_hist, on="Datetime", how="left")
        merged_df["Source"] = "Historical"

        # Forecast climate
        next_hours = [start_datetime + timedelta(hours=i) for i in range(1, 168+1)]
        forecast_df = pd.DataFrame({"Datetime": next_hours})
        forecast_start, forecast_end = forecast_df["Datetime"].min(), forecast_df["Datetime"].max()

        if forecast_end < gmt7_now:
            climate_fore = fetch_climate_historical(forecast_start, forecast_end)
        elif forecast_start > gmt7_now:
            climate_fore = fetch_climate_forecast()
        else:
            hist = fetch_climate_historical(forecast_start, gmt7_now)
            fore = fetch_climate_forecast()
            climate_fore = pd.concat([hist, fore]).drop_duplicates(subset="Datetime")

        forecast_df = pd.merge(forecast_df, climate_fore, on="Datetime", how="left")
        forecast_df["Water_level"] = np.nan
        forecast_df["Source"] = "Forecast"

        # -----------------------------
        # Prepare full_df for prediction
        # -----------------------------
        full_df = pd.concat([merged_df, forecast_df], ignore_index=True).sort_values("Datetime")
        full_df.reset_index(drop=True, inplace=True)

        # Function to build lagged features
        def build_lag_features(df, current_time):
            feature = {}
            # Water level lags 1-24
            for lag in range(1,25):
                lag_time = current_time - pd.Timedelta(hours=lag)
                mask = df["Datetime"] == lag_time
                feature[f"Water_level_Lag{lag}"] = df.loc[mask, "Water_level"].iloc[0] if mask.any() else np.nan
            # Rainfall lags 17-24
            for lag in range(17,25):
                lag_time = current_time - pd.Timedelta(hours=lag)
                mask = df["Datetime"] == lag_time
                feature[f"Rainfall_Lag{lag}"] = df.loc[mask, "Rainfall"].iloc[0] if mask.any() and "Rainfall" in df.columns else np.nan
            # Cloud_cover lags 1-24
            for lag in range(1,25):
                lag_time = current_time - pd.Timedelta(hours=lag)
                mask = df["Datetime"] == lag_time
                feature[f"Cloud_cover_Lag{lag}"] = df.loc[mask, "Cloud_cover"].iloc[0] if mask.any() and "Cloud_cover" in df.columns else np.nan
            # Surface_pressure lags 1-24
            for lag in range(1,25):
                lag_time = current_time - pd.Timedelta(hours=lag)
                mask = df["Datetime"] == lag_time
                feature[f"Surface_pressure_Lag{lag}"] = df.loc[mask, "Surface_pressure"].iloc[0] if mask.any() and "Surface_pressure" in df.columns else np.nan
            # Soil_temperature lags 9-11
            for lag in range(9,12):
                lag_time = current_time - pd.Timedelta(hours=lag)
                mask = df["Datetime"] == lag_time
                feature[f"Soil_temperature_Lag{lag}"] = df.loc[mask, "Soil_temperature"].iloc[0] if mask.any() and "Soil_temperature" in df.columns else np.nan
            # Soil_moisture lags 1-24
            for lag in range(1,25):
                lag_time = current_time - pd.Timedelta(hours=lag)
                mask = df["Datetime"] == lag_time
                feature[f"Soil_moisture_Lag{lag}"] = df.loc[mask, "Soil_moisture"].iloc[0] if mask.any() and "Soil_moisture" in df.columns else np.nan
            return feature

        # -----------------------------
        # Sequential prediction 7x24 hours
        # -----------------------------
        forecast_hours = full_df.loc[full_df["Source"]=="Forecast","Datetime"]
        for current_time in forecast_hours:
            feature_dict = build_lag_features(full_df, current_time)
            feature_df = pd.DataFrame([feature_dict])
            pred = model.predict(feature_df)[0]
            full_df.loc[full_df["Datetime"]==current_time, "Water_level"] = round(pred,2)

        # -----------------------------
        # Display final dataframe
        # -----------------------------
        st.subheader("Water Level + Climate Data + Forecast")
        display_cols = ["Datetime","Water_level","Rainfall","Cloud_cover","Surface_pressure","Soil_temperature","Soil_moisture","Source"]
        st.dataframe(full_df[display_cols], use_container_width=True, height=500)
