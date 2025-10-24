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
feature_cols = model.get_booster().feature_names  # kolom yang dipakai model

st.title("🌊 Water Level Forecast Dashboard")

# -----------------------------
# Current time (GMT+7), rounded up
# -----------------------------
now_utc = datetime.utcnow()
gmt7_now = now_utc + timedelta(hours=7)
rounded_now = (gmt7_now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0) \
    if (gmt7_now.minute > 0 or gmt7_now.second > 0) else gmt7_now.replace(minute=0, second=0, microsecond=0)

# -----------------------------
# Select start datetime
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
# Fetch climate functions
# -----------------------------
def fetch_climate_historical(start_dt, end_dt, lat=-0.117, lon=114.1):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_dt.date()}&end_date={end_dt.date()}"
        f"&hourly=rain,surface_pressure,cloud_cover,soil_temperature_0_to_7cm,soil_moisture_0_to_7cm"
        f"&timezone=Asia%2FBangkok"
    )
    try:
        data = requests.get(url, timeout=30).json()
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
        data = requests.get(url, timeout=30).json()
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
# Build lagged features
# -----------------------------
def build_lag_features(df, current_time, max_lags=24):
    lagged = {}
    # Rainfall lag17-24
    for lag in range(17, 25):
        lag_time = current_time - timedelta(hours=lag)
        val = df.loc[df["Datetime"] == lag_time, "Rainfall"]
        lagged[f"Rainfall_Lag{lag}"] = val.values[0] if not val.empty else 0
    # Cloud_cover lag1-24
    for lag in range(1, 25):
        lag_time = current_time - timedelta(hours=lag)
        val = df.loc[df["Datetime"] == lag_time, "Cloud_cover"]
        lagged[f"Cloud_cover_Lag{lag}"] = val.values[0] if not val.empty else 0
    # Surface_pressure lag1-24
    for lag in range(1, 25):
        lag_time = current_time - timedelta(hours=lag)
        val = df.loc[df["Datetime"] == lag_time, "Surface_pressure"]
        lagged[f"Surface_pressure_Lag{lag}"] = val.values[0] if not val.empty else 0
    # Soil_temperature lag9-11
    for lag in range(9, 12):
        lag_time = current_time - timedelta(hours=lag)
        val = df.loc[df["Datetime"] == lag_time, "Soil_temperature"]
        lagged[f"Soil_temperature_Lag{lag}"] = val.values[0] if not val.empty else 0
    # Soil_moisture lag1-24
    for lag in range(1, 25):
        lag_time = current_time - timedelta(hours=lag)
        val = df.loc[df["Datetime"] == lag_time, "Soil_moisture"]
        lagged[f"Soil_moisture_Lag{lag}"] = val.values[0] if not val.empty else 0
    # Water_level lag1-24
    for lag in range(1, 25):
        lag_time = current_time - timedelta(hours=lag)
        val = df.loc[df["Datetime"] == lag_time, "Water_level"]
        lagged[f"Water_level_Lag{lag}"] = val.values[0] if not val.empty else 0
    return lagged

# -----------------------------
# Fetch Data & Forecasting
# -----------------------------
if wl_hourly is not None:
    if st.button("Fetch Climate Data & Perform Forecasting"):

        with st.spinner("Fetching Climate Data and Performing Forecast..."):
            start_dt = wl_hourly["Datetime"].min()
            end_dt = wl_hourly["Datetime"].max()
            climate_df = fetch_climate_historical(start_dt, end_dt)

            merged_df = pd.merge(
                wl_hourly, climate_df, on="Datetime", how="left"
            ).sort_values(by="Datetime", ascending=True)

            # Generate next 7x24 hours
            next_hours = [start_datetime + timedelta(hours=i) for i in range(1, 168 + 1)]
            forecast_df = pd.DataFrame({"Datetime": next_hours})
            forecast_start, forecast_end = forecast_df["Datetime"].min(), forecast_df["Datetime"].max()

            # Determine source of climate data
            if forecast_end < gmt7_now:
                add_df = fetch_climate_historical(forecast_start, forecast_end)
            elif forecast_start > gmt7_now:
                add_df = fetch_climate_forecast()
            else:
                hist_df = fetch_climate_historical(forecast_start, gmt7_now)
                fore_df = fetch_climate_forecast()
                add_df = pd.concat([hist_df, fore_df]).drop_duplicates(subset="Datetime")

            forecast_merged = pd.merge(forecast_df, add_df, on="Datetime", how="left")
            forecast_merged["Water_level"] = np.nan
            forecast_merged["Source"] = "Forecast"

            merged_df["Source"] = "Historical"

            full_df = pd.concat([merged_df, forecast_merged], ignore_index=True).sort_values(by="Datetime", ascending=True)

            # -----------------------------
            # Predict water level bertahap per jam
            # -----------------------------
            feature_cols = [c for c in full_df.columns if "_Lag" in c]
            for i in range(len(forecast_merged)):
                current_time = forecast_merged.loc[i, "Datetime"]
                feature_dict = {}

                # Ambil lag feature dari full_df (observed + pred sebelumnya)
                for col in feature_cols:
                    base_name, lag_num = col.rsplit("_Lag", 1)
                    lag_num = int(lag_num)
                    lag_time = current_time - pd.Timedelta(hours=lag_num)
                    if lag_time in full_df["Datetime"].values:
                        val = full_df.loc[full_df["Datetime"] == lag_time, base_name]
                        if not val.empty:
                            feature_dict[col] = val.values[0]
                        else:
                            feature_dict[col] = 0.0
                    else:
                        feature_dict[col] = 0.0

                feature_df = pd.DataFrame([feature_dict])
                pred = model.predict(feature_df)[0]
                pred = max(pred, 0)  # minimal 0

                # Masukkan hasil prediksi ke full_df
                full_df.loc[full_df["Datetime"] == current_time, "Water_level"] = pred

            # -----------------------------
            # Tampilkan tabel
            # -----------------------------
            # Round numeric columns
            for col in full_df.select_dtypes(include=np.number).columns:
                full_df[col] = full_df[col].round(2)

            # Highlight forecast rows biru
            def highlight_forecast(row):
                color = 'background-color: #cfe9ff' if row['Source'] == 'Forecast' else ''
                return [color] * len(row)

            styled_df = full_df.style.apply(highlight_forecast, axis=1).format("{:.2f}")

            st.subheader("Water Level + Climate Data + Forecast")
            st.dataframe(styled_df, use_container_width=True, height=500)
