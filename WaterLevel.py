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
Fetch_climate_data + Fetch & Predict
# -----------------------------
def fetch_climate_data(start_dt, end_dt, mode="historical"):
    """
    Ambil data iklim dari Open-Meteo dan kembalikan DataFrame dengan kolom:
    Datetime (naive GMT+7 floored), Rainfall, Cloud_cover, Surface_pressure,
    Soil_moisture, Soil_temperature.

    start_dt, end_dt can be python datetime or pandas Timestamp (naive GMT+7).
    """
    try:
        # normalisasi ke pandas Timestamp (naive)
        start_ts = pd.Timestamp(start_dt).tz_localize(None)
        end_ts = pd.Timestamp(end_dt).tz_localize(None)

        if mode == "historical":
            start_date = start_ts.date().isoformat()
            end_date = end_ts.date().isoformat()
            url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude=-0.117&longitude=114.1"
                f"&start_date={start_date}&end_date={end_date}"
                f"&hourly=rain,cloud_cover,surface_pressure,soil_moisture_0_to_7cm,soil_temperature_0_to_7cm"
                f"&timezone=Asia%2FSingapore"
            )
        else:
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude=-0.117&longitude=114.1"
                f"&hourly=rain,cloud_cover,surface_pressure,soil_moisture_0_to_1cm,soil_temperature_0cm"
                f"&timezone=Asia%2FSingapore&forecast_days=14"
            )

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        hourly = j.get("hourly", {})
        if not hourly or "time" not in hourly:
            st.warning(f"No hourly data returned for {mode}")
            return pd.DataFrame(columns=["Datetime","Rainfall","Cloud_cover","Surface_pressure","Soil_moisture","Soil_temperature"])

        df = pd.DataFrame(hourly)

        # rename fields to unified names if present
        rename_map = {}
        if "rain" in df.columns: rename_map["rain"] = "Rainfall"
        if "cloud_cover" in df.columns: rename_map["cloud_cover"] = "Cloud_cover"
        if "surface_pressure" in df.columns: rename_map["surface_pressure"] = "Surface_pressure"
        if "soil_moisture_0_to_1cm" in df.columns: rename_map["soil_moisture_0_to_1cm"] = "Soil_moisture"
        if "soil_moisture_0_to_7cm" in df.columns: rename_map["soil_moisture_0_to_7cm"] = "Soil_moisture"
        if "soil_temperature_0cm" in df.columns: rename_map["soil_temperature_0cm"] = "Soil_temperature"
        if "soil_temperature_0_to_7cm" in df.columns: rename_map["soil_temperature_0_to_7cm"] = "Soil_temperature"
        df = df.rename(columns=rename_map)

        # parse times -> API returns local times; convert to Timestamp, add 0 or +7h if needed
        # We'll convert to Timestamp and floor to hour. (If times already local GMT+7, adding +7h would double-shift.)
        # Many endpoints return strings like "2025-10-23T00:00" without tz; treat them as Asia/Singapore local
        df["Datetime"] = pd.to_datetime(df["time"])
        # floor hours and ensure naive
        df["Datetime"] = df["Datetime"].dt.floor("H").dt.tz_localize(None)

        # ensure columns exist
        cols = ["Datetime", "Rainfall", "Cloud_cover", "Surface_pressure", "Soil_moisture", "Soil_temperature"]
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA

        # filter using Timestamp comparisons (start_ts, end_ts)
        df = df.loc[(df["Datetime"] >= start_ts) & (df["Datetime"] < end_ts), cols].reset_index(drop=True)
        return df

    except Exception as e:
        st.warning(f"fetch_climate_data({mode}) error: {e}")
        return pd.DataFrame(columns=["Datetime","Rainfall","Cloud_cover","Surface_pressure","Soil_moisture","Soil_temperature"])


# -----------------------------
# Actual Fetch & Predict block (fixed)
# -----------------------------
if st.button("Fetch & Predict 🌦️") and wl_hourly is not None:
    try:
        st.info("📡 Mengambil data iklim dan melakukan forecasting - Mohon Tunggu...")
        total_hours = 7 * 24

        # Normalize start to pandas Timestamp (naive GMT+7)
        start_ts = pd.Timestamp(start_datetime).tz_localize(None)
        end_ts = start_ts + pd.Timedelta(hours=total_hours)
        now_gmt7 = (pd.Timestamp.utcnow() + pd.Timedelta(hours=7)).tz_localize(None)

        # historical range: at least 24h before start up to now (if before end)
        hist_start = start_ts - pd.Timedelta(hours=24)
        hist_end = min(end_ts, now_gmt7)

        df_hist = pd.DataFrame()
        if hist_end > hist_start:
            df_hist = fetch_climate_data(hist_start, hist_end, mode="historical")

        df_fore = pd.DataFrame()
        if end_ts > now_gmt7:
            fc_start = max(now_gmt7, start_ts)
            df_fore = fetch_climate_data(fc_start, end_ts, mode="forecast")

        # combine and dedupe
        df_weather = pd.concat([df_hist, df_fore], ignore_index=True) if (not df_hist.empty or not df_fore.empty) else pd.DataFrame()
        if not df_weather.empty:
            df_weather = df_weather.drop_duplicates(subset="Datetime", keep="first").sort_values("Datetime").reset_index(drop=True)

        # prepare prediction timeline (list of pandas Timestamps)
        forecast_hours = [start_ts + pd.Timedelta(hours=i) for i in range(total_hours)]
        df_pred = pd.DataFrame({"Datetime": forecast_hours})
        if not df_weather.empty:
            df_pred = df_pred.merge(df_weather, on="Datetime", how="left")
        else:
            # ensure columns exist
            df_pred["Rainfall"] = pd.NA
            df_pred["Cloud_cover"] = pd.NA
            df_pred["Surface_pressure"] = pd.NA
            df_pred["Soil_moisture"] = pd.NA
            df_pred["Soil_temperature"] = pd.NA

        df_pred = df_pred.set_index("Datetime")

        # normalize wl_hourly Datetime and map actual 24h values
        wl_hourly["Datetime"] = pd.to_datetime(wl_hourly["Datetime"]).dt.tz_localize(None)
        wl_dict = dict(zip(wl_hourly["Datetime"], wl_hourly["Water_level"]))
        df_pred["Water_level_actual"] = df_pred.index.map(wl_dict)

        # prepare lag feature order
        lag_features = []
        for i in range(17, 25): lag_features.append(f"Rainfall_Lag{i}")
        for i in range(1, 25):  lag_features.append(f"Cloud_cover_Lag{i}")
        for i in range(1, 25):  lag_features.append(f"Surface_pressure_Lag{i}")
        for i in range(9, 12):  lag_features.append(f"Soil_temperature_Lag{i}")
        for i in range(1, 25):  lag_features.append(f"Soil_moisture_Lag{i}")
        for i in range(1, 25):  lag_features.append(f"Water_level_Lag{i}")

        # helper to safely read weather cell (by Timestamp index)
        def _safe_weather_lookup(ts, col):
            try:
                v = df_pred.at[ts, col]
                return float(v) if pd.notna(v) else 0.0
            except Exception:
                return 0.0

        # initial water level lags from wl_hourly (use keys relative to start_ts)
        water_level_lags = []
        for i in range(1, 25):
            key = start_ts - pd.Timedelta(hours=i)
            water_level_lags.append(float(wl_dict.get(key, 0.0)))

        # iterative prediction
        df_pred["Water_level_pred"] = pd.NA
        for ts in df_pred.index:
            # build input dictionary using pd.Timedelta offsets
            inp = {}
            for i in range(17, 25):
                t_lag = ts - pd.Timedelta(hours=i)
                inp[f"Rainfall_Lag{i}"] = [_safe_weather_lookup(t_lag, "Rainfall")]
            for i in range(1, 25):
                t_lag = ts - pd.Timedelta(hours=i)
                inp[f"Cloud_cover_Lag{i}"] = [_safe_weather_lookup(t_lag, "Cloud_cover")]
                inp[f"Surface_pressure_Lag{i}"] = [_safe_weather_lookup(t_lag, "Surface_pressure")]
                inp[f"Soil_moisture_Lag{i}"] = [_safe_weather_lookup(t_lag, "Soil_moisture")]
                inp[f"Water_level_Lag{i}"] = [water_level_lags[i-1]]
            for i in range(9, 12):
                t_lag = ts - pd.Timedelta(hours=i)
                inp[f"Soil_temperature_Lag{i}"] = [_safe_weather_lookup(t_lag, "Soil_temperature")]

            input_df = pd.DataFrame(inp)[lag_features].fillna(0.0)

            pred_val = float(model.predict(input_df)[0])
            df_pred.at[ts, "Water_level_pred"] = round(pred_val, 3)

            # if actual exists for this ts (uploaded), use actual; otherwise use pred for next steps
            actual_val = df_pred.at[ts, "Water_level_actual"] if "Water_level_actual" in df_pred.columns else pd.NA
            if pd.notna(actual_val):
                next_val = float(actual_val)
            else:
                next_val = pred_val

            # update rolling water_level_lags
            water_level_lags = [next_val] + water_level_lags[:-1]

        # combine into single column
        df_pred["Water_level"] = np.where(pd.notna(df_pred["Water_level_actual"]), df_pred["Water_level_actual"], df_pred["Water_level_pred"])

        # determine last actual time for highlighting
        last_actual_time = None
        if not wl_hourly.empty:
            last_actual_time = pd.to_datetime(wl_hourly["Datetime"]).dt.tz_localize(None).max()

        # preview dataframe
        preview = df_pred.reset_index()[["Datetime", "Water_level", "Rainfall", "Cloud_cover", "Surface_pressure", "Soil_moisture", "Soil_temperature"]]

        def highlight_predicted(row):
            if last_actual_time is pd.NaT or last_actual_time is None:
                return ['' for _ in row]
            return ['background-color: #FFF3B0' if row["Datetime"] > last_actual_time else '' for _ in row]

        st.subheader("Preview (Actual & Predicted per hour)")
        st.dataframe(preview.style.apply(highlight_predicted, axis=1).format({
            "Water_level": "{:.3f}",
            "Rainfall": "{:.3f}",
            "Cloud_cover": "{:.3f}",
            "Surface_pressure": "{:.3f}",
            "Soil_moisture": "{:.3f}",
            "Soil_temperature": "{:.3f}"
        }), height=520, use_container_width=True)

        # plot combined
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=preview["Datetime"], y=preview["Water_level"], mode="lines+markers", name="Water Level (actual+pred)", line=dict(color="royalblue")))
        if last_actual_time is not None:
            fig.add_vline(x=last_actual_time, line_dash="dash", line_color="red", annotation_text="Prediction Start", annotation_position="top right")
        fig.update_layout(title=f"Water Level (Actual + Predicted) from {start_ts.strftime('%Y-%m-%d %H:%M')} (GMT+7)", xaxis_title="Datetime (GMT+7)", yaxis_title="Water Level (m)", hovermode="x unified", height=540)
        st.plotly_chart(fig, use_container_width=True)

        st.success("✅ Fetch & Predict selesai.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ Fetch & Predict selesai.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
