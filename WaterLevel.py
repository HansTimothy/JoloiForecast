import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np
from datetime import datetime, timedelta, time

# -----------------------------
# Load trained XGBoost model
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
                .sort_values(by="Datetime", ascending=True)
                .round(2)
            )

            st.success("Successfully uploaded 24-hour water level data before start time.")
            st.dataframe(wl_hourly)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

# -----------------------------
# Functions to fetch climate data
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
# Run 7-day iterative forecast
# -----------------------------
if wl_hourly is not None:
    if st.button("Run 7-Day Water Level Forecast"):
        with st.spinner("Fetching climate data and performing 7-day forecast..."):
            # --- Merge historical water level with climate data ---
            climate_hist = fetch_climate_historical(wl_hourly["Datetime"].min(), wl_hourly["Datetime"].max())
            merged_df = pd.merge(wl_hourly, climate_hist, on="Datetime", how="left").sort_values("Datetime")

            # --- Prepare forecast dataframe for next 7 days ---
            next_hours = [start_datetime + timedelta(hours=i) for i in range(1, 7*24+1)]
            forecast_df = pd.DataFrame({"Datetime": next_hours})
            forecast_climate = fetch_climate_forecast()
            forecast_df = pd.merge(forecast_df, forecast_climate, on="Datetime", how="left")
            forecast_df["Water_level"] = np.nan
            forecast_df["Source"] = "Forecast"

            merged_df["Source"] = "Historical"
            full_df = pd.concat([merged_df, forecast_df], ignore_index=True).sort_values("Datetime").reset_index(drop=True)

            # --- Iterative forecast per hour using last 24h (historical + forecasted) ---
            lag_hours = 24
            for i in range(len(full_df)):
                if full_df.at[i, "Source"] != "Forecast":
                    continue  # skip historical
                if i < lag_hours:
                    continue  # skip if not enough history
                last_24h = full_df.loc[i-lag_hours:i-1, "Water_level"].values.reshape(1, -1)
                try:
                    full_df.at[i, "Water_level"] = model.predict(last_24h)[0]
                except:
                    full_df.at[i, "Water_level"] = np.nan

            # --- Round numeric columns ---
            full_df = full_df.apply(lambda x: np.round(x,2) if np.issubdtype(x.dtype, np.number) else x)

            # --- Display result ---
            st.subheader("Water Level + Climate Data (7-Day Forecast)")
            def highlight_forecast(row):
                return ['background-color: #cfe9ff' if row['Source']=="Forecast" else '' for _ in row]
            format_dict = {col: "{:.2f}" for col in full_df.select_dtypes(include=np.number).columns}
            styled_df = full_df.style.apply(highlight_forecast, axis=1).format(format_dict)
            st.dataframe(styled_df, use_container_width=True, height=500)
