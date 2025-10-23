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
# Fetch climate helper + Actual Fetch & Predict (perbaikan tipe waktu)
# -----------------------------
import numpy as np

def fetch_climate_data(start_dt, end_dt, mode="historical"):
    """
    Ambil data iklim dari Open-Meteo dan kembalikan DataFrame dengan kolom:
    Datetime (naive GMT+7 floored), Rainfall, Cloud_cover, Surface_pressure,
    Soil_moisture, Soil_temperature.

    start_dt, end_dt : bisa berupa python datetime atau pandas Timestamp (naive GMT+7)
    mode: "historical" atau "forecast"
    """
    try:
        # normalisasi input ke pandas Timestamp (naive)
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

        # Rename columns to unified names if present
        rename_map = {}
        if "rain" in df.columns: rename_map["rain"] = "Rainfall"
        if "cloud_cover" in df.columns: rename_map["cloud_cover"] = "Cloud_cover"
        if "surface_pressure" in df.columns: rename_map["surface_pressure"] = "Surface_pressure"
        if "soil_moisture_0_to_1cm" in df.columns: rename_map["soil_moisture_0_to_1cm"] = "Soil_moisture"
        if "soil_moisture_0_to_7cm" in df.columns: rename_map["soil_moisture_0_to_7cm"] = "Soil_moisture"
        if "soil_temperature_0cm" in df.columns: rename_map["soil_temperature_0cm"] = "Soil_temperature"
        if "soil_temperature_0_to_7cm" in df.columns: rename_map["soil_temperature_0_to_7cm"] = "Soil_temperature"

        df = df.rename(columns=rename_map)

        # Parse time -> to pandas Timestamps, then convert to naive GMT+7 by adding 7 hours (API returns timezone Asia/Singapore)
        # Note: API returns time strings already in timezone Asia/Singapore when timezone param used, but to be safe we add +7h then floor.
        df["Datetime"] = pd.to_datetime(df["time"]) + pd.Timedelta(hours=7)
        df["Datetime"] = df["Datetime"].dt.floor("H").dt.tz_localize(None)

        # Ensure unified columns exist
        cols = ["Datetime", "Rainfall", "Cloud_cover", "Surface_pressure", "Soil_moisture", "Soil_temperature"]
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA

        # Filter by requested interval [start_ts, end_ts)
        df = df.loc[(df["Datetime"] >= start_ts) & (df["Datetime"] < end_ts), cols].reset_index(drop=True)
        return df

    except Exception as e:
        st.warning(f"fetch_climate_data({mode}) error: {e}")
        return pd.DataFrame(columns=["Datetime","Rainfall","Cloud_cover","Surface_pressure","Soil_moisture","Soil_temperature"])


# -----------------------------
# Actual Fetch & Predict block (perbaikan tipe waktu)
# -----------------------------
if st.button("Fetch & Predict 🌦️") and wl_hourly is not None:
    try:
        st.info("📡 Mengambil data iklim...")
        total_hours = 7 * 24
        # normalize start_datetime to pandas Timestamp naive
        start_ts = pd.Timestamp(start_datetime).tz_localize(None)
        end_ts = start_ts + pd.Timedelta(hours=total_hours)
        now_gmt7 = pd.Timestamp.utcnow() + pd.Timedelta(hours=7)
        now_gmt7 = now_gmt7.tz_localize(None)

        # historical range: we need at least 24h before start for water-level lag and up to min(end_ts, now)
        hist_start = start_ts - pd.Timedelta(hours=24)
        hist_end = min(end_ts, now_gmt7)

        # fetch historical (archive) for needed range [hist_start, hist_end)
        df_hist = pd.DataFrame()
        if hist_end > hist_start:
            df_hist = fetch_climate_data(hist_start, hist_end, mode="historical")

        # fetch forecast only for future window that historical doesn't cover
        df_fore = pd.DataFrame()
        if end_ts > now_gmt7:
            fc_start = max(now_gmt7, start_ts)
            df_fore = fetch_climate_data(fc_start, end_ts, mode="forecast")

        # combine weather data (prioritize historical for overlapping timestamps)
        df_weather = pd.concat([df_hist, df_fore], ignore_index=True)
        if not df_weather.empty:
            df_weather = df_weather.drop_duplicates(subset="Datetime", keep="first").sort_values("Datetime").reset_index(drop=True)
        st.success(f"Data iklim siap — total records: {len(df_weather)}")

        # prepare prediction frame for all hours from start_datetime
        forecast_hours = [ (start_ts + pd.Timedelta(hours=i)) for i in range(total_hours) ]
        df_pred = pd.DataFrame({"Datetime": forecast_hours})
        # merge weather (left join, so df_pred keeps all hours)
        if not df_weather.empty:
            df_pred = df_pred.merge(df_weather, on="Datetime", how="left")
        else:
            # ensure weather columns exist
            df_pred["Rainfall"] = pd.NA
            df_pred["Cloud_cover"] = pd.NA
            df_pred["Surface_pressure"] = pd.NA
            df_pred["Soil_moisture"] = pd.NA
            df_pred["Soil_temperature"] = pd.NA

        df_pred = df_pred.set_index("Datetime")

        # map uploaded water level (which is already naive GMT+7 floored) to df_pred
        wl_hourly["Datetime"] = pd.to_datetime(wl_hourly["Datetime"]).dt.tz_localize(None)
        wl_dict = dict(zip(wl_hourly["Datetime"], wl_hourly["Water_level"]))
        df_pred["Water_level_actual"] = df_pred.index.map(wl_dict)

        # prepare lag feature order (same order as your model)
        lag_features = []
        for i in range(17, 25): lag_features.append(f"Rainfall_Lag{i}")
        for i in range(1, 25):  lag_features.append(f"Cloud_cover_Lag{i}")
        for i in range(1, 25):  lag_features.append(f"Surface_pressure_Lag{i}")
        for i in range(9, 12):  lag_features.append(f"Soil_temperature_Lag{i}")
        for i in range(1, 25):  lag_features.append(f"Soil_moisture_Lag{i}")
        for i in range(1, 25):  lag_features.append(f"Water_level_Lag{i}")

        # helper to safely pick value from df_pred (weather cols) by datetime
        def _safe_weather_lookup(idx, col):
            try:
                v = df_pred.at[idx, col]
                return float(v) if pd.notna(v) else 0.0
            except Exception:
                return 0.0

        # initial water level lags (lag1..lag24): use uploaded wl or 0.0 if missing
        water_level_lags = []
        for i in range(1, 25):
            key = start_ts - pd.Timedelta(hours=i)
            water_level_lags.append(float(wl_dict.get(key, 0.0)))

        # iterative per-hour prediction
        df_pred["Water_level_pred"] = pd.NA
        st.info("Forecasting — mohon tunggu...")
        for dt in df_pred.index:
            # construct input dict
            inp = {}
            # Rainfall 17..24
            for i in range(17, 25):
                t = dt - pd.Timedelta(hours=i)
                inp[f"Rainfall_Lag{i}"] = [ _safe_weather_lookup(t, "Rainfall") ]
            # Cloud_cover 1..24
            for i in range(1, 25):
                t = dt - pd.Timedelta(hours=i)
                inp[f"Cloud_cover_Lag{i}"] = [ _safe_weather_lookup(t, "Cloud_cover") ]
            # Surface_pressure 1..24
            for i in range(1, 25):
                t = dt - pd.Timedelta(hours=i)
                inp[f"Surface_pressure_Lag{i}"] = [ _safe_weather_lookup(t, "Surface_pressure") ]
            # Soil_temperature 9..11
            for i in range(9, 12):
                t = dt - pd.Timedelta(hours=i)
                inp[f"Soil_temperature_Lag{i}"] = [ _safe_weather_lookup(t, "Soil_temperature") ]
            # Soil_moisture 1..24
            for i in range(1, 25):
                t = dt - pd.Timedelta(hours=i)
                inp[f"Soil_moisture_Lag{i}"] = [ _safe_weather_lookup(t, "Soil_moisture") ]
            # Water_level 1..24 (from rolling list)
            for i in range(1, 25):
                inp[f"Water_level_Lag{i}"] = [ water_level_lags[i - 1] ]

            # create input dataframe in the same order as features
            input_df = pd.DataFrame(inp)[lag_features].fillna(0.0)

            # predict
            pred_val = float(model.predict(input_df)[0])
            df_pred.at[dt, "Water_level_pred"] = round(pred_val, 3)

            # if there is an actual observed value for this dt (uploaded), prefer that as 'value to use' for next lags
            actual = df_pred.at[dt, "Water_level_actual"] if "Water_level_actual" in df_pred.columns else pd.NA
            if pd.notna(actual):
                value_to_use = float(actual)
            else:
                value_to_use = pred_val

            # update rolling water_level_lags
            water_level_lags = [float(value_to_use)] + water_level_lags[:-1]

        # combine actual & predicted into single column
        df_pred["Water_level"] = np.where(
            pd.notna(df_pred["Water_level_actual"]),
            df_pred["Water_level_actual"],
            df_pred["Water_level_pred"]
        )

        # last actual time (for highlight / vline)
        last_actual_time = wl_hourly["Datetime"].max()

        # prepare preview dataframe (reset index to get Datetime column)
        preview = df_pred.reset_index()[[
            "Datetime", "Water_level",
            "Rainfall", "Cloud_cover", "Surface_pressure",
            "Soil_moisture", "Soil_temperature"
        ]]

        # highlight predicted rows (Datetime > last_actual_time)
        def highlight_predicted(row):
            if pd.isna(last_actual_time):
                return ['' for _ in row]
            return ['background-color: #FFF3B0' if row["Datetime"] > last_actual_time else '' for _ in row]

        st.subheader("Preview (Actual & Predicted per hour)")
        st.dataframe(
            preview.style.apply(highlight_predicted, axis=1).format({
                "Water_level": "{:.3f}",
                "Rainfall": "{:.3f}",
                "Cloud_cover": "{:.3f}",
                "Surface_pressure": "{:.3f}",
                "Soil_moisture": "{:.3f}",
                "Soil_temperature": "{:.3f}"
            }),
            height=480,
            use_container_width=True
        )

        # plot combined line
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=preview["Datetime"],
            y=preview["Water_level"],
            mode="lines+markers",
            name="Water Level (actual + pred)",
            line=dict(color="royalblue")
        ))

        if pd.notna(last_actual_time):
            fig.add_vline(
                x=last_actual_time,
                line_dash="dash",
                line_color="red",
                annotation_text="Prediction Start",
                annotation_position="top right"
            )

        fig.update_layout(
            title=f"Water Level (Actual + Predicted) from {start_ts.strftime('%Y-%m-%d %H:%M')} (GMT+7)",
            xaxis_title="Datetime (GMT+7)",
            yaxis_title="Water Level (m)",
            hovermode="x unified",
            height=540
        )

        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ Fetch & Predict selesai.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
