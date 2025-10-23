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
# Actual Fetch & Predict block (fixed + safe)
# -----------------------------
if st.button("Fetch & Predict 🌦️") and wl_hourly is not None:
    try:
        st.info("📡 Mengambil data iklim dan melakukan forecasting - mohon tunggu...")

        start_ts = pd.Timestamp(start_datetime).tz_localize(None)
        total_hours = 7 * 24
        end_ts = start_ts + pd.Timedelta(hours=total_hours)
        now_gmt7 = (pd.Timestamp.utcnow() + pd.Timedelta(hours=7)).tz_localize(None)

        # -----------------------------
        # 1) Fetch climate data
        # -----------------------------
        hist_start = start_ts - pd.Timedelta(hours=24)
        hist_end = min(end_ts, now_gmt7)
        df_hist = pd.DataFrame()
        if hist_end > hist_start:
            df_hist = fetch_climate_data(hist_start, hist_end, mode="historical")

        df_fore = pd.DataFrame()
        if end_ts > now_gmt7:
            fc_start = max(now_gmt7, start_ts)
            df_fore = fetch_climate_data(fc_start, end_ts, mode="forecast")

        # Combine
        if not df_hist.empty and not df_fore.empty:
            df_weather = pd.concat([df_hist, df_fore], ignore_index=True)
        elif not df_hist.empty:
            df_weather = df_hist.copy()
        elif not df_fore.empty:
            df_weather = df_fore.copy()
        else:
            df_weather = pd.DataFrame(columns=["Datetime","Rainfall","Cloud_cover","Surface_pressure","Soil_moisture","Soil_temperature"])

        if not df_weather.empty:
            df_weather["Datetime"] = pd.to_datetime(df_weather["Datetime"]).dt.floor("H").dt.tz_localize(None)
            df_weather = df_weather.drop_duplicates("Datetime").sort_values("Datetime").reset_index(drop=True)

        # -----------------------------
        # 2) Build prediction timeline
        # -----------------------------
        timeline = pd.date_range(start=start_ts, periods=total_hours, freq="H")
        df_pred = pd.DataFrame({"Datetime": timeline})
        if not df_weather.empty:
            df_pred = df_pred.merge(df_weather, on="Datetime", how="left")
        else:
            for c in ["Rainfall","Cloud_cover","Surface_pressure","Soil_moisture","Soil_temperature"]:
                df_pred[c] = pd.NA
        df_pred = df_pred.set_index("Datetime")

        # -----------------------------
        # 3) Prepare water level lags
        # -----------------------------
        wl_hourly["Datetime"] = pd.to_datetime(wl_hourly["Datetime"]).dt.floor("H").dt.tz_localize(None)
        wl_map = pd.Series(wl_hourly["Water_level"].values, index=wl_hourly["Datetime"]).to_dict()
        df_pred["Water_level_actual"] = df_pred.index.map(wl_map)

        # Lag feature order
        lag_features = []
        for i in range(17,25): lag_features.append(f"Rainfall_Lag{i}")
        for i in range(1,25): 
            lag_features += [f"Cloud_cover_Lag{i}", f"Surface_pressure_Lag{i}", f"Soil_moisture_Lag{i}", f"Water_level_Lag{i}"]
        for i in range(9,12): lag_features.append(f"Soil_temperature_Lag{i}")

        # Initial water level lags (24 hours before start)
        water_level_lags = [float(wl_map.get(start_ts - pd.Timedelta(hours=h), 0.0)) for h in range(1,25)]

        # Helper to safely lookup weather
        def safe_weather(ts, col):
            try:
                val = df_pred.at[ts, col]
                return float(val) if pd.notna(val) else 0.0
            except:
                return 0.0

        # -----------------------------
        # 4) Iterative prediction
        # -----------------------------
        df_pred["Water_level_pred"] = pd.NA
        for ts in df_pred.index:
            inp = {}
            # Rainfall lags 17-24
            for i in range(17,25):
                lag_ts = ts - pd.Timedelta(hours=i)
                inp[f"Rainfall_Lag{i}"] = [ safe_weather(lag_ts, "Rainfall") ]
            # Cloud, Pressure, Soil moisture lags 1-24 + Water level lags
            for i in range(1,25):
                lag_ts = ts - pd.Timedelta(hours=i)
                inp[f"Cloud_cover_Lag{i}"] = [ safe_weather(lag_ts, "Cloud_cover") ]
                inp[f"Surface_pressure_Lag{i}"] = [ safe_weather(lag_ts, "Surface_pressure") ]
                inp[f"Soil_moisture_Lag{i}"] = [ safe_weather(lag_ts, "Soil_moisture") ]
                inp[f"Water_level_Lag{i}"] = [ water_level_lags[i-1] ]
            # Soil temperature lags 9-11
            for i in range(9,12):
                lag_ts = ts - pd.Timedelta(hours=i)
                inp[f"Soil_temperature_Lag{i}"] = [ safe_weather(lag_ts, "Soil_temperature") ]

            input_df = pd.DataFrame(inp)[lag_features].fillna(0.0)
            pred_val = float(model.predict(input_df)[0])
            df_pred.at[ts, "Water_level_pred"] = round(pred_val,3)

            # Roll water_level_lags
            actual_val = df_pred.at[ts, "Water_level_actual"] if pd.notna(df_pred.at[ts,"Water_level_actual"]) else None
            value_to_use = float(actual_val) if actual_val is not None else pred_val
            water_level_lags = [value_to_use] + water_level_lags[:-1]

        # -----------------------------
        # 5) Combine actual & predicted
        # -----------------------------
        df_pred["Water_level"] = np.where(pd.notna(df_pred["Water_level_actual"]), df_pred["Water_level_actual"], df_pred["Water_level_pred"])
        last_actual_time = wl_hourly["Datetime"].max() if not wl_hourly.empty else None

        # -----------------------------
        # 6) Preview & highlight
        # -----------------------------
        preview = df_pred.reset_index()[["Datetime","Water_level","Rainfall","Cloud_cover","Surface_pressure","Soil_moisture","Soil_temperature"]]
        def highlight_pred(row):
            if last_actual_time is None: return ['' for _ in row]
            return ['background-color:#FFF3B0' if row["Datetime"] > last_actual_time else '' for _ in row]

        st.subheader("Preview (Actual & Predicted per hour)")
        st.dataframe(preview.style.apply(highlight_pred, axis=1).format({
            "Water_level":"{:.3f}",
            "Rainfall":"{:.3f}",
            "Cloud_cover":"{:.3f}",
            "Surface_pressure":"{:.3f}",
            "Soil_moisture":"{:.3f}",
            "Soil_temperature":"{:.3f}"
        }), height=520, use_container_width=True)

        # -----------------------------
        # 7) Plot
        # -----------------------------
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=preview["Datetime"], y=preview["Water_level"], mode="lines+markers", name="Water Level (Actual+Pred)", line=dict(color="royalblue")))
        if last_actual_time is not None:
            fig.add_vline(x=last_actual_time, line_dash="dash", line_color="red", annotation_text="Prediction Start", annotation_position="top right")
        fig.update_layout(title=f"Water Level (Actual + Predicted) from {start_ts.strftime('%Y-%m-%d %H:%M')} (GMT+7)",
                          xaxis_title="Datetime (GMT+7)", yaxis_title="Water Level (m)", hovermode="x unified", height=540)
        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ Fetch & Predict selesai.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
