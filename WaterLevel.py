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

st.title("Water Level Forecast Dashboard 🌊")

# -----------------------------
# Waktu sekarang (GMT+7) rounded up
# -----------------------------
now_utc = datetime.utcnow()  # UTC
gmt7_now = now_utc + timedelta(hours=7)

# Round up ke jam
if gmt7_now.minute > 0 or gmt7_now.second > 0 or gmt7_now.microsecond > 0:
    rounded_now = (gmt7_now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
else:
    rounded_now = gmt7_now.replace(minute=0, second=0, microsecond=0)

# -----------------------------
# Pilih datetime start forecast
# -----------------------------
st.subheader("Pilih Tanggal & Jam Mulai 7-Day Forecast")

selected_date = st.date_input(
    "Tanggal",
    value=rounded_now.date(),
    max_value=rounded_now.date()
)

hour_options = [f"{h:02d}:00" for h in range(0, rounded_now.hour + 1)]
selected_hour_str = st.selectbox(
    "Jam",
    hour_options,
    index=len(hour_options)-1
)

selected_hour = int(selected_hour_str.split(":")[0])

# Gabungkan menjadi naive datetime
start_datetime = datetime.combine(selected_date, time(selected_hour, 0, 0))
st.write(f"Start datetime (GMT+7): {start_datetime}")

# -----------------------------
# Upload water level file
# -----------------------------
st.subheader("Upload Water Level File (Hourly)")
uploaded_file = st.file_uploader("Upload file CSV AWLR Logs Joloi", type=["csv"])
wl_hourly = None
if uploaded_file is not None:
    try:
        df_wl = pd.read_csv(uploaded_file, engine='python', skip_blank_lines=True)
        if "Datetime" not in df_wl.columns or "Level Air" not in df_wl.columns:
            st.error("File harus memiliki kolom 'Datetime' dan 'Level Air'")
        else:
            # Konversi ke datetime naive
            df_wl["Datetime"] = pd.to_datetime(df_wl["Datetime"]).dt.floor("H")
            
            # Filter 24 jam sebelum start_datetime
            start_limit = start_datetime - pd.Timedelta(hours=24)
            df_wl_filtered = df_wl[(df_wl["Datetime"] >= start_limit) & (df_wl["Datetime"] < start_datetime)]

            # Group by jam dan ambil rata-rata
            wl_hourly = df_wl_filtered.groupby("Datetime")["Level Air"].mean().reset_index()
            wl_hourly.rename(columns={"Level Air": "Water_level"}, inplace=True)
            wl_hourly = wl_hourly.sort_values(by="Datetime", ascending=False)

            st.success("Data water level 24 jam sebelum start berhasil diupload")
            st.dataframe(wl_hourly.style.format({"Water_level":"{:.2f}"}))
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")

# -----------------------------
# Fungsi fetch climate historis
# -----------------------------
def fetch_climate_historical(start_dt, end_dt, lat=-0.117, lon=114.1):
    start_date = start_dt.date().isoformat()
    end_date = end_dt.date().isoformat()
    
    st.info(f"Fetching climate data (historical) from {start_date} to {end_date}")

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
            "Surface_pressure": data["hourly"]["surface_pressure"],
            "Cloud_cover": data["hourly"]["cloud_cover"],
            "Soil_temperature": data["hourly"]["soil_temperature_0_to_7cm"],
            "Soil_moisture": data["hourly"]["soil_moisture_0_to_7cm"],
            "Rainfall": data["hourly"]["rain"]
        })
        df["Datetime"] = df["Datetime"].dt.floor("H")
        return df
    except Exception as e:
        st.error(f"Gagal fetch climate data historis: {e}")
        return pd.DataFrame()

# -----------------------------
# Fungsi fetch climate forecast
# -----------------------------
def fetch_climate_forecast(lat=-0.117, lon=114.1):
    st.info("Fetching climate data (forecast) 14 hari ke depan...")
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
            "Surface_pressure": data["hourly"]["surface_pressure"],
            "Cloud_cover": data["hourly"]["cloud_cover"],
            "Soil_temperature": data["hourly"]["soil_temperature_0cm"],
            "Soil_moisture": data["hourly"]["soil_moisture_0_to_1cm"],
            "Rainfall": data["hourly"]["rain"]
        })
        df["Datetime"] = df["Datetime"].dt.floor("H")
        return df
    except Exception as e:
        st.error(f"Gagal fetch climate forecast: {e}")
        return pd.DataFrame()

# -----------------------------
# Merge dengan water level + tambah 7x24 jam forecast
# -----------------------------
if wl_hourly is not None:
    if st.button("Fetch Climate Data & Extend 7x24 Jam"):
        start_dt = wl_hourly["Datetime"].min()
        end_dt = wl_hourly["Datetime"].max()
        climate_df = fetch_climate_historical(start_dt, end_dt)

        # Merge historis 24 jam terakhir
        merged_df = pd.merge(wl_hourly, climate_df, on="Datetime", how="left")
        merged_df = merged_df.sort_values(by="Datetime", ascending=False)

        # 7x24 jam ke depan
        next_hours = [start_datetime + timedelta(hours=i) for i in range(1, 168 + 1)]
        forecast_df = pd.DataFrame({"Datetime": next_hours})

        forecast_start = forecast_df["Datetime"].min()
        forecast_end = forecast_df["Datetime"].max()

        # Tentukan apakah historis / forecast / campuran
        if forecast_end < gmt7_now:
            # Semua historis
            add_df = fetch_climate_historical(forecast_start, forecast_end)
        elif forecast_start > gmt7_now:
            # Semua forecast
            add_df = fetch_climate_forecast()
        else:
            # Campuran
            hist_df = fetch_climate_historical(forecast_start, gmt7_now)
            fore_df = fetch_climate_forecast()
            add_df = pd.concat([hist_df, fore_df]).drop_duplicates(subset="Datetime")

        # Gabungkan dengan forecast_df (pastikan hanya 168 jam)
        forecast_merged = pd.merge(forecast_df, add_df, on="Datetime", how="left")
        forecast_merged["Water_level"] = np.nan  # placeholder
        forecast_merged["Source"] = "Forecast 7x24"

        merged_df["Source"] = "Observed"

        final_df = pd.concat([merged_df, forecast_merged], ignore_index=True)
        final_df = final_df.sort_values(by="Datetime", ascending=False)

        # Highlight biru untuk 7x24 jam
        def highlight_blue(row):
            color = "background-color: lightblue" if row["Source"] == "Forecast 7x24" else ""
            return [color] * len(row)

        st.subheader("Merged Water Level + Climate Data (Extended 7x24 Jam)")
        st.dataframe(final_df.round(2).style.apply(highlight_blue, axis=1))
