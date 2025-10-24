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
    if gmt7_now.minute > 0 or gmt7_now.second > 0 or gmt7_now.microsecond > 0 \
    else gmt7_now.replace(minute=0, second=0, microsecond=0)

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
    if "Datetime" in df_wl.columns and "Level Air" in df_wl.columns:
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
    else:
        st.error("The file must contain columns 'Datetime' and 'Level Air'.")

# -----------------------------
# Climate data fetch functions
# -----------------------------
def fetch_climate_historical(start_dt, end_dt, lat=-0.117, lon=114.1):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_dt.date().isoformat()}&end_date={end_dt.date().isoformat()}"
        f"&hourly=surface_pressure,cloud_cover,soil_temperature_0_to_7cm,soil_moisture_0_to_7cm,rain"
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
    except:
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
    except:
        return pd.DataFrame()

# -----------------------------
# Run Fetch + Forecast
# -----------------------------
if wl_hourly is not None:
    if st.button("Fetch Climate Data & Run 7-Day Forecast"):
        with st.spinner("Fetching climate data and performing 7-day forecast..."):
            
            # 1️⃣ Merge historical water level + climate
            start_dt, end_dt = wl_hourly["Datetime"].min(), wl_hourly["Datetime"].max()
            climate_hist = fetch_climate_historical(start_dt, end_dt)
            merged_df = pd.merge(wl_hourly, climate_hist, on="Datetime", how="left").sort_values("Datetime")

            # 2️⃣ Create forecast frame for next 7 days
            forecast_hours = 168
            forecast_dates = [start_datetime + timedelta(hours=i) for i in range(1, forecast_hours+1)]
            forecast_df = pd.DataFrame({"Datetime": forecast_dates})
            forecast_start, forecast_end = forecast_df["Datetime"].min(), forecast_df["Datetime"].max()

            # Fetch climate for forecast
            if forecast_end < gmt7_now:
                add_df = fetch_climate_historical(forecast_start, forecast_end)
            elif forecast_start > gmt7_now:
                add_df = fetch_climate_forecast()
            else:
                hist_df = fetch_climate_historical(forecast_start, gmt7_now)
                fore_df = fetch_climate_forecast()
                add_df = pd.concat([hist_df, fore_df]).drop_duplicates(subset="Datetime")

            forecast_df = pd.merge(forecast_df, add_df, on="Datetime", how="left")
            forecast_df["Water_level"] = np.nan
            forecast_df["Source"] = "Forecast"

            merged_df["Source"] = "Historical"
            full_df = pd.concat([merged_df, forecast_df], ignore_index=True).sort_values("Datetime")

            # 3️⃣ Define highlighted features for lag
            highlighted_features = [
                "Rainfall_Lag17","Rainfall_Lag18","Rainfall_Lag19","Rainfall_Lag20","Rainfall_Lag21","Rainfall_Lag22","Rainfall_Lag23","Rainfall_Lag24",
                "Cloud_cover_Lag1","Cloud_cover_Lag2","Cloud_cover_Lag3","Cloud_cover_Lag4","Cloud_cover_Lag5","Cloud_cover_Lag6","Cloud_cover_Lag7",
                "Surface_pressure_Lag1","Surface_pressure_Lag2","Surface_pressure_Lag3","Surface_pressure_Lag4","Surface_pressure_Lag5","Surface_pressure_Lag6",
                "Soil_temperature_Lag9","Soil_temperature_Lag10","Soil_temperature_Lag11",
                "Soil_moisture_Lag1","Soil_moisture_Lag2","Soil_moisture_Lag3","Soil_moisture_Lag4","Soil_moisture_Lag5","Soil_moisture_Lag6",
                "Water_level_Lag1","Water_level_Lag2","Water_level_Lag3","Water_level_Lag4","Water_level_Lag5"
            ]

            # 4️⃣ Rolling forecast
            df_pred = full_df.copy()
            for i in range(len(df_pred)):
                if df_pred.loc[i, "Source"] == "Forecast":
                    current_dt = df_pred.loc[i, "Datetime"]
                    lag_data = {}
                    for feat in highlighted_features:
                        base, lag_num = feat.rsplit("_Lag",1)
                        lag_num = int(lag_num)
                        lag_time = current_dt - timedelta(hours=lag_num)
                        if base == "Water_level":
                            val = df_pred.loc[df_pred["Datetime"]==lag_time, "Water_level"].values
                        else:
                            val = df_pred.loc[df_pred["Datetime"]==lag_time, base].values
                        lag_data[feat] = val[0] if len(val)>0 else np.nan

                    X_pred = pd.DataFrame([lag_data])[highlighted_features]
                    df_pred.loc[i, "Water_level"] = model.predict(X_pred)[0] if not X_pred.isna().any().any() else np.nan

            # 5️⃣ Display final DataFrame
            st.subheader("Water Level + Climate Data with Forecast")
            styled_df = df_pred.style.format("{:.2f}").apply(
                lambda row: ['background-color: #cfe9ff' if row['Source']=="Forecast" else '' for _ in row], axis=1
            )
            st.dataframe(styled_df, use_container_width=True, height=500)
