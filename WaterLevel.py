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

            st.success("Data water level 24 jam sebelum start berhasil diupload")
            st.dataframe(wl_hourly.style.format({"Water_level":"{:.2f}"}))
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")

# -----------------------------
# Tombol fetch data historis
# -----------------------------
st.subheader("Fetch Climate Data")

if wl_hourly is not None:
    if st.button("Fetch Climate Data"):
        st.info("Mengambil data iklim...")

        # -----------------------------
        # Fungsi fetch data cuaca historis
        # -----------------------------
        def fetch_climate_data(start_dt, end_dt):
            """
            Ambil data iklim dari Open-Meteo API
            start_dt, end_dt: datetime (naive GMT+7)
            return: DataFrame dengan kolom Datetime, Rainfall, Cloud_cover, Surface_pressure
            """
            # Konversi ke ISO format UTC (Open-Meteo default UTC)
            start_utc = (start_dt - timedelta(hours=7)).isoformat()
            end_utc = (end_dt - timedelta(hours=7)).isoformat()

            url = (
                f"https://archive-api.open-meteo.com/v1/archive?latitude=-0.117&longitude=114.1"
                f"&start_date={start_utc}&end_date={end_utc}"
                f"&hourly=temperature_2m,surface_pressure,cloud_cover,soil_temperature_0_to_7cm,soil_moisture_0_to_7cm"
                f"&timezone=Asia%2FBangkok"
            )

            try:
                response = requests.get(url)
                data = response.json()
                
                df = pd.DataFrame({
                    "Datetime": pd.to_datetime(data["hourly"]["time"]),
                    "Rainfall": data["hourly"]["precipitation"],
                    "Cloud_cover": data["hourly"]["cloudcover"],
                    "Surface_pressure": data["hourly"]["surface_pressure"]
                })

                # Konversi ke GMT+7
                df["Datetime"] = df["Datetime"] + timedelta(hours=7)
                df["Datetime"] = df["Datetime"].dt.floor("H")

                return df
            except Exception as e:
                st.error(f"Gagal fetch data cuaca: {e}")
                return None

        # -----------------------------
        # Ambil data cuaca 24 jam terakhir sebelum start_datetime
        # -----------------------------
        start_limit = start_datetime - timedelta(hours=24)
        climate_df = fetch_climate_data(start_limit, start_datetime)

        if climate_df is not None:
            # Merge dengan water level
            merged_df = pd.merge(wl_hourly, climate_df, on="Datetime", how="left")
            st.success("Data cuaca berhasil diambil dan digabungkan")
            st.dataframe(merged_df.style.format({
                "Water_level":"{:.2f}",
                "Rainfall":"{:.2f}",
                "Cloud_cover":"{:.2f}",
                "Surface_pressure":"{:.2f}"
            }))
        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ Fetch & Predict selesai.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
