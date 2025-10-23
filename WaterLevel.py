# WaterLevel_API_hourly_v2.py
import streamlit as st
import pandas as pd
import requests
import joblib
import plotly.graph_objects as go
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
            # ubah ke datetime naive dan floor jam
            df_wl["Datetime"] = pd.to_datetime(df_wl["Datetime"]) + timedelta(hours=7)  # GMT+7
            df_wl["Datetime"] = df_wl["Datetime"].dt.floor("H")  # Naive
            df_wl["Datetime"] = df_wl["Datetime"].dt.tz_localize(None)

            # filter 24 jam sebelum start_datetime
            start_limit = start_datetime - timedelta(hours=24)
            df_wl = df_wl[(df_wl["Datetime"] >= start_limit) & (df_wl["Datetime"] < start_datetime)]
            
            # group by jam
            wl_hourly = df_wl.groupby("Datetime")["Level Air"].mean().reset_index()
            wl_hourly.rename(columns={"Level Air": "Water_level"}, inplace=True)
            
            st.success(f"Data water level berhasil diupload")
            st.dataframe(wl_hourly.style.format({"Water_level":"{:.2f}"}))
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")

# -----------------------------
# Fetch data & predict
# -----------------------------
# -----------------------------
# Fetch data & predict
# -----------------------------
if st.button("Fetch Weather & Predict") and wl_hourly is not None:
    try:
        st.info("Mengambil data iklim dari API...")

        # Durasi prediksi: 7 hari (168 jam)
        end_datetime = start_datetime + timedelta(hours=7 * 24)
        now_gmt7 = datetime.utcnow() + timedelta(hours=7)

        # Tentukan rentang historis dan forecast
        hist_end = min(end_datetime, now_gmt7)
        forecast_start = max(start_datetime, now_gmt7)

        # Ambil data historis iklim
        df_hist = fetch_climate_data(start_datetime - timedelta(hours=24), hist_end, mode="historical")

        # Ambil data forecast iklim jika diperlukan
        if forecast_start < end_datetime:
            df_forecast = fetch_climate_data(forecast_start, end_datetime, mode="forecast")
            df_weather = pd.concat([df_hist, df_forecast], ignore_index=True)
        else:
            df_weather = df_hist.copy()

        df_weather = df_weather.sort_values("Datetime").reset_index(drop=True)
        st.success(f"Data iklim berhasil diambil ({len(df_weather)} jam)")

        # -----------------------------
        # Siapkan dataframe prediksi
        # -----------------------------
        forecast_hours = [start_datetime + timedelta(hours=i) for i in range(7 * 24)]
        df_pred = pd.DataFrame({"Datetime": forecast_hours})
        df_pred = df_pred.merge(df_weather, on="Datetime", how="left")

        # Masukkan water level historis (dari upload)
        wl_dict = dict(zip(wl_hourly["Datetime"], wl_hourly["Water_level"]))
        df_pred["Water_level"] = df_pred["Datetime"].map(wl_dict)

        df_pred.set_index("Datetime", inplace=True)

        # -----------------------------
        # Daftar fitur lag
        # -----------------------------
        lag_features = []
        for i in range(17, 25): lag_features.append(f"Rainfall_Lag{i}")
        for i in range(1, 25): lag_features.append(f"Cloud_cover_Lag{i}")
        for i in range(1, 25): lag_features.append(f"Surface_pressure_Lag{i}")
        for i in range(9, 12): lag_features.append(f"Soil_temperature_Lag{i}")
        for i in range(1, 25): lag_features.append(f"Soil_moisture_Lag{i}")
        for i in range(1, 25): lag_features.append(f"Water_level_Lag{i}")

        # -----------------------------
        # Helper function
        # -----------------------------
        def safe_get(df, dt, col):
            try:
                return float(df.loc[dt, col])
            except:
                return 0.0

        # Inisialisasi lag water level (24 jam terakhir dari data upload)
        water_level_lags = [safe_get(df_pred, dt, "Water_level") for dt in wl_hourly["Datetime"][-24:]]

        # -----------------------------
        # Prediksi bertahap per jam
        # -----------------------------
        st.info("Melakukan prediksi per jam selama 7 hari...")
        results = {}

        for dt in df_pred.index:
            inp = {}

            # Rainfall lag 17–24
            for i in range(17, 25):
                inp[f"Rainfall_Lag{i}"] = [safe_get(df_pred, dt - timedelta(hours=i), "Rainfall")]

            # Cloud cover lag 1–24
            for i in range(1, 25):
                inp[f"Cloud_cover_Lag{i}"] = [safe_get(df_pred, dt - timedelta(hours=i), "Cloud_cover")]

            # Surface pressure lag 1–24
            for i in range(1, 25):
                inp[f"Surface_pressure_Lag{i}"] = [safe_get(df_pred, dt - timedelta(hours=i), "Surface_pressure")]

            # Soil temperature lag 9–11
            for i in range(9, 12):
                inp[f"Soil_temperature_Lag{i}"] = [safe_get(df_pred, dt - timedelta(hours=i), "Soil_temperature")]

            # Soil moisture lag 1–24
            for i in range(1, 25):
                inp[f"Soil_moisture_Lag{i}"] = [safe_get(df_pred, dt - timedelta(hours=i), "Soil_moisture")]

            # Water level lag 1–24 (pakai prediksi sebelumnya)
            for i in range(1, 25):
                inp[f"Water_level_Lag{i}"] = [water_level_lags[i - 1]]

            input_data = pd.DataFrame(inp)[lag_features].fillna(0.0)

            # Prediksi model
            pred = model.predict(input_data)[0]
            df_pred.loc[dt, "Predicted_Water_level"] = round(pred, 2)
            results[dt] = pred

            # Update lag water level untuk jam berikutnya
            water_level_lags = [pred] + water_level_lags[:-1]

        # -----------------------------
        # Tampilkan hasil
        # -----------------------------
        st.subheader("Hasil Prediksi (Preview 7 Hari / 168 Jam)")
        df_result = df_pred.reset_index()[["Datetime", "Predicted_Water_level"]]
        st.dataframe(df_result.style.format({"Predicted_Water_level": "{:.2f}"}), height=400)

        # -----------------------------
        # Plot hasil
        # -----------------------------
        st.subheader("Grafik Prediksi Water Level")
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_result["Datetime"],
            y=df_result["Predicted_Water_level"],
            mode="lines+markers",
            name="Predicted Water Level",
            line=dict(color="blue", width=2)
        ))

        if wl_hourly is not None:
            fig.add_trace(go.Scatter(
                x=wl_hourly["Datetime"],
                y=wl_hourly["Water_level"],
                mode="lines+markers",
                name="Observed Water Level (Input)",
                line=dict(color="green", width=2, dash="dot")
            ))

        fig.update_layout(
            xaxis_title="Datetime (GMT+7)",
            yaxis_title="Water Level (m)",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
