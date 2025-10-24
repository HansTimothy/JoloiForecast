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
rounded_now = (gmt7_now + timedelta(hours=1) if gmt7_now.minute > 0 else gmt7_now).replace(minute=0, second=0, microsecond=0)

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
# Climate fetch functions
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
# Run 7-day forecast
# -----------------------------
if wl_hourly is not None and st.button("Run 7-Day Forecast"):

    # -----------------------------
    # Prepare historical + climate
    # -----------------------------
    start_dt = wl_hourly["Datetime"].min()
    end_dt = wl_hourly["Datetime"].max()
    climate_hist = fetch_climate_historical(start_dt, end_dt)
    merged_df = pd.merge(wl_hourly, climate_hist, on="Datetime", how="left").sort_values("Datetime")
    merged_df["Source"] = "Historical"

    # -----------------------------
    # Prepare forecast df
    # -----------------------------
    forecast_hours = [start_datetime + timedelta(hours=i) for i in range(168)]
    forecast_df = pd.DataFrame({"Datetime": forecast_hours})

    if forecast_hours[-1] < gmt7_now:
        climate_forecast = fetch_climate_historical(forecast_hours[0], forecast_hours[-1])
    elif forecast_hours[0] > gmt7_now:
        climate_forecast = fetch_climate_forecast()
    else:
        climate_forecast = pd.concat([
            fetch_climate_historical(forecast_hours[0], gmt7_now),
            fetch_climate_forecast()
        ]).drop_duplicates(subset="Datetime")

    forecast_merged = pd.merge(forecast_df, climate_forecast, on="Datetime", how="left")
    forecast_merged["Water_level"] = np.nan
    forecast_merged["Source"] = "Forecast"

    final_df = pd.concat([merged_df, forecast_merged], ignore_index=True).sort_values("Datetime").reset_index(drop=True)

    # -----------------------------
    # Iterative prediction
    # -----------------------------
    # Ambil list kolom model
    model_features = model.get_booster().feature_names  # pastikan nama kolom sama persis dengan saat training

    for idx, row in final_df[final_df["Source"]=="Forecast"].iterrows():
        # Ambil last 24 jam water level + climate lagged sesuai feature
        df_window = final_df.loc[idx-24:idx-1].copy()  # pastikan ada 24 baris sebelumnya
        if df_window.shape[0] < 24:
            continue  # skip jika belum cukup data
        X_pred = pd.DataFrame(columns=model_features, index=[0])
        for f in model_features:
            base, lag = f.rsplit("_Lag", 1)
            lag = int(lag)
            if base == "Water_level":
                X_pred.at[0,f] = final_df.loc[idx-lag,"Water_level"]
            else:
                X_pred.at[0,f] = final_df.loc[idx-lag,base]
        y_hat = model.predict(X_pred)[0]
        final_df.at[idx,"Water_level"] = round(y_hat,2)

    # -----------------------------
    # Display in Streamlit
    # -----------------------------
    st.subheader("Water Level + Climate Data with Forecast")
    def highlight_forecast(row):
        return ['background-color: #cfe9ff' if row['Source']=="Forecast" else '' for _ in row]

    format_dict = {col: "{:.2f}" for col in final_df.select_dtypes(include=np.number).columns}
    st.dataframe(final_df.style.apply(highlight_forecast, axis=1).format(format_dict), use_container_width=True, height=500)
