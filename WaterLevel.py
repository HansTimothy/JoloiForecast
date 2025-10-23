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
# Fetch climate helper + Fetch & Predict block
# -----------------------------
def fetch_climate_data(start_dt, end_dt, mode="historical"):
    """
    Ambil data iklim dari Open-Meteo.
    - start_dt, end_dt: naive datetimes (GMT+7), inclusive start, exclusive end
    - mode: "historical" atau "forecast"
    Return: DataFrame dengan kolom:
      Datetime, Rainfall, Cloud_cover, Surface_pressure, Soil_moisture, Soil_temperature
    Datetime dikembalikan sebagai naive GMT+7 floored to hour.
    """
    try:
        if mode == "historical":
            start_date = start_dt.strftime("%Y-%m-%d")
            end_date = end_dt.strftime("%Y-%m-%d")
            url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude=-0.117&longitude=114.1"
                f"&start_date={start_date}&end_date={end_date}"
                f"&hourly=rain,cloud_cover,surface_pressure,soil_moisture_0_to_7cm,soil_temperature_0_to_7cm"
                f"&timezone=Asia%2FSingapore"
            )
        else:  # forecast
            # forecast_days set to 14 as you requested
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
        # normalize column names that can differ between endpoint versions
        # map API fields to our unified names
        rename_map = {}
        if "rain" in df.columns: rename_map["rain"] = "Rainfall"
        if "cloud_cover" in df.columns: rename_map["cloud_cover"] = "Cloud_cover"
        if "surface_pressure" in df.columns: rename_map["surface_pressure"] = "Surface_pressure"
        # soil moisture/temp columns differ between archive and forecast:
        if "soil_moisture_0_to_1cm" in df.columns:
            rename_map["soil_moisture_0_to_1cm"] = "Soil_moisture"
        if "soil_moisture_0_to_7cm" in df.columns:
            rename_map["soil_moisture_0_to_7cm"] = "Soil_moisture"
        if "soil_temperature_0cm" in df.columns:
            rename_map["soil_temperature_0cm"] = "Soil_temperature"
        if "soil_temperature_0_to_7cm" in df.columns:
            rename_map["soil_temperature_0_to_7cm"] = "Soil_temperature"

        df = df.rename(columns=rename_map)

        # parse time -> convert to GMT+7 naive by adding 7 hours then floor to hour
        df["Datetime"] = pd.to_datetime(df["time"]) + timedelta(hours=7)
        df["Datetime"] = df["Datetime"].dt.floor("H").dt.tz_localize(None)

        # Keep only the unified columns (if missing, create with NaN)
        cols = ["Datetime", "Rainfall", "Cloud_cover", "Surface_pressure", "Soil_moisture", "Soil_temperature"]
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA

        df = df[cols]
        # filter by requested datetimes (start inclusive, end exclusive)
        df = df[(df["Datetime"] >= start_dt) & (df["Datetime"] < end_dt)].reset_index(drop=True)
        return df

    except Exception as e:
        st.warning(f"fetch_climate_data({mode}) error: {e}")
        return pd.DataFrame(columns=["Datetime","Rainfall","Cloud_cover","Surface_pressure","Soil_moisture","Soil_temperature"])


# -----------------------------
# Actual Fetch & Predict block (versi gabungan actual + predicted)
# -----------------------------
if st.button("Fetch & Predict 🌦️") and wl_hourly is not None:
    try:
        st.info("📡 Mengambil data iklim...")
        total_hours = 7 * 24
        end_datetime = start_datetime + timedelta(hours=total_hours)
        now_gmt7 = datetime.utcnow() + timedelta(hours=7)

        # Ambil data historis untuk 24 jam sebelum start
        hist_start = start_datetime - timedelta(hours=24)
        hist_end = min(end_datetime, now_gmt7)

        df_hist = pd.DataFrame()
        if hist_end > hist_start:
            df_hist = fetch_climate_data(hist_start, hist_end, mode="historical")

        # Ambil forecast untuk waktu mendatang (jika ada)
        df_fore = pd.DataFrame()
        if end_datetime > now_gmt7:
            fc_start = max(now_gmt7, start_datetime)
            df_fore = fetch_climate_data(fc_start, end_datetime, mode="forecast")

        # Gabungkan data iklim
        df_weather = pd.concat([df_hist, df_fore], ignore_index=True)
        df_weather = df_weather.drop_duplicates(subset="Datetime", keep="first").sort_values("Datetime").reset_index(drop=True)
        st.success(f"Data iklim siap — total records: {len(df_weather)}")

        # Siapkan dataframe prediksi
        forecast_hours = [start_datetime + timedelta(hours=i) for i in range(total_hours)]
        df_pred = pd.DataFrame({"Datetime": forecast_hours})
        df_pred = df_pred.merge(df_weather, on="Datetime", how="left")
        df_pred = df_pred.set_index("Datetime")

        # Map actual water level 24 jam terakhir
        wl_dict = dict(zip(wl_hourly["Datetime"], wl_hourly["Water_level"]))
        df_pred["Water_level_actual"] = df_pred.index.map(wl_dict)
        df_pred["Water_level_pred"] = pd.NA

        # Urutan fitur lag
        lag_features = []
        for i in range(17, 25): lag_features.append(f"Rainfall_Lag{i}")
        for i in range(1, 25):  lag_features.append(f"Cloud_cover_Lag{i}")
        for i in range(1, 25):  lag_features.append(f"Surface_pressure_Lag{i}")
        for i in range(9, 12):  lag_features.append(f"Soil_temperature_Lag{i}")
        for i in range(1, 25):  lag_features.append(f"Soil_moisture_Lag{i}")
        for i in range(1, 25):  lag_features.append(f"Water_level_Lag{i}")

        # Fungsi bantu ambil nilai cuaca aman
        def safe_weather(dt, col):
            try:
                val = df_pred.at[dt, col]
                return float(val) if pd.notna(val) else 0.0
            except Exception:
                return 0.0

        # Siapkan lag awal water level
        water_level_lags = []
        for i in range(1, 25):
            key = start_datetime - timedelta(hours=i)
            water_level_lags.append(float(wl_dict.get(key, 0.0)))

        st.info("🔮 Melakukan prediksi bertahap...")
        for dt in df_pred.index:
            # Input model
            inp = {}
            for i in range(17, 25):
                inp[f"Rainfall_Lag{i}"] = [safe_weather(dt - timedelta(hours=i), "Rainfall")]
            for i in range(1, 25):
                inp[f"Cloud_cover_Lag{i}"] = [safe_weather(dt - timedelta(hours=i), "Cloud_cover")]
                inp[f"Surface_pressure_Lag{i}"] = [safe_weather(dt - timedelta(hours=i), "Surface_pressure")]
                inp[f"Soil_moisture_Lag{i}"] = [safe_weather(dt - timedelta(hours=i), "Soil_moisture")]
                inp[f"Water_level_Lag{i}"] = [water_level_lags[i - 1]]
            for i in range(9, 12):
                inp[f"Soil_temperature_Lag{i}"] = [safe_weather(dt - timedelta(hours=i), "Soil_temperature")]

            X_pred = pd.DataFrame(inp)[lag_features].fillna(0.0)
            y_pred = float(model.predict(X_pred)[0])
            df_pred.at[dt, "Water_level_pred"] = round(y_pred, 3)

            # Update rolling lags
            actual_val = df_pred.at[dt, "Water_level_actual"]
            next_val = float(actual_val) if pd.notna(actual_val) else y_pred
            water_level_lags = [next_val] + water_level_lags[:-1]

        # -----------------------------
        # Gabungkan Actual + Prediksi jadi satu kolom
        # -----------------------------
        df_pred["Water_level"] = np.where(
            pd.notna(df_pred["Water_level_actual"]),
            df_pred["Water_level_actual"],
            df_pred["Water_level_pred"]
        )

        last_actual_time = wl_hourly["Datetime"].max()

        # Preview tabel
        preview = df_pred.reset_index()[[
            "Datetime", "Water_level",
            "Rainfall", "Cloud_cover", "Surface_pressure",
            "Soil_moisture", "Soil_temperature"
        ]]

        def highlight_predicted(row):
            return [
                'background-color: #FFF3B0' if row["Datetime"] > last_actual_time else ''
                for _ in row
            ]

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

        # -----------------------------
        # Plot gabungan actual + predicted
        # -----------------------------
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=preview["Datetime"],
            y=preview["Water_level"],
            mode="lines+markers",
            name="Water Level (Actual + Predicted)",
            line=dict(color="royalblue")
        ))

        fig.add_vline(
            x=last_actual_time,
            line_dash="dash",
            line_color="red",
            annotation_text="Prediction Start",
            annotation_position="top right"
        )

        fig.update_layout(
            title=f"Water Level (Actual & Predicted) from {start_datetime.strftime('%Y-%m-%d %H:%M')} (GMT+7)",
            xaxis_title="Datetime (GMT+7)",
            yaxis_title="Water Level (m)",
            hovermode="x unified",
            height=540
        )

        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ Fetch & Predict selesai.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
