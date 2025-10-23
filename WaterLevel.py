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
# Actual Fetch & Predict block
# -----------------------------
# -----------------------------
# Fetch data & predict
# -----------------------------
if st.button("Fetch & Predict 🌦️"):
    if wl_hourly is None:
        st.error("Harap upload file water level terlebih dahulu.")
    else:
        try:
            st.info("📡 Mengambil data iklim dari API... Mohon tunggu.")

            # ============================================
            # 1. Tentukan periode data
            # ============================================
            now_gmt7 = datetime.utcnow() + timedelta(hours=7)
            end_datetime = start_datetime + timedelta(days=7)
            start_limit = start_datetime - timedelta(hours=24)

            use_forecast = end_datetime > now_gmt7  # Jika ada periode ke depan
            end_for_archive = min(end_datetime, now_gmt7)

            # ============================================
            # 2. Fungsi ambil data iklim
            # ============================================
            def fetch_climate_data(start_dt, end_dt, mode="historical"):
                if mode == "historical":
                    url = (
                        f"https://archive-api.open-meteo.com/v1/archive?"
                        f"latitude=-0.117&longitude=114.1"
                        f"&start_date={start_dt.date()}&end_date={end_dt.date()}"
                        f"&hourly=rain,cloud_cover,surface_pressure,"
                        f"soil_temperature_0_to_7cm,soil_moisture_0_to_7cm"
                        f"&timezone=Asia%2FSingapore"
                    )
                else:
                    url = (
                        f"https://api.open-meteo.com/v1/forecast?"
                        f"latitude=-0.117&longitude=114.1"
                        f"&hourly=rain,cloud_cover,surface_pressure,"
                        f"soil_moisture_0_to_1cm,soil_temperature_0cm"
                        f"&timezone=Asia%2FSingapore&forecast_days=14"
                    )

                r = requests.get(url)
                r.raise_for_status()
                data = r.json()

                df = pd.DataFrame({
                    "Datetime": pd.to_datetime(data["hourly"]["time"]),
                    "Rainfall": data["hourly"]["rain"],
                    "Cloud_cover": data["hourly"]["cloud_cover"],
                    "Surface_pressure": data["hourly"]["surface_pressure"],
                    "Soil_temperature": (
                        data["hourly"]["soil_temperature_0_to_7cm"]
                        if "soil_temperature_0_to_7cm" in data["hourly"]
                        else data["hourly"]["soil_temperature_0cm"]
                    ),
                    "Soil_moisture": (
                        data["hourly"]["soil_moisture_0_to_7cm"]
                        if "soil_moisture_0_to_7cm" in data["hourly"]
                        else data["hourly"]["soil_moisture_0_to_1cm"]
                    ),
                })
                df["Datetime"] = df["Datetime"].dt.tz_localize(None)
                return df

            # ============================================
            # 3. Ambil data iklim
            # ============================================
            climate_historical = fetch_climate_data(start_limit, end_for_archive, "historical")

            if use_forecast:
                climate_forecast = fetch_climate_data(now_gmt7, end_datetime, "forecast")
                climate_all = pd.concat([climate_historical, climate_forecast]).drop_duplicates("Datetime")
            else:
                climate_all = climate_historical

            # ============================================
            # 4. Gabungkan dengan data water level actual
            # ============================================
            df_all = pd.merge(climate_all, wl_hourly, on="Datetime", how="left")
            df_all["Source"] = df_all["Water_level"].apply(lambda x: "Actual" if pd.notnull(x) else "Predicted")

            # ============================================
            # 5. Fungsi buat lag features
            # ============================================
            def create_lagged_features(df):
                lag_features = []
                for i in range(17,25): lag_features.append(f"Rainfall_Lag{i}")
                for i in range(1,25): lag_features.append(f"Cloud_cover_Lag{i}")
                for i in range(1,25): lag_features.append(f"Surface_pressure_Lag{i}")
                for i in range(9,12): lag_features.append(f"Soil_temperature_Lag{i}")
                for i in range(1,25): lag_features.append(f"Soil_moisture_Lag{i}")
                for i in range(1,25): lag_features.append(f"Water_level_Lag{i}")

                for feature in lag_features:
                    base, lag = feature.split("_Lag")
                    df[feature] = df[base].shift(int(lag))
                return df, lag_features

            # ============================================
            # 6. Prediksi bertahap per jam (auto-lag)
            # ============================================
            df_all = df_all.sort_values("Datetime").reset_index(drop=True)
            df_all, lag_features = create_lagged_features(df_all)

            preds = []
            for i in range(len(df_all)):
                row = df_all.iloc[i]
                if pd.isna(row["Water_level"]) and all(f in df_all.columns for f in lag_features):
                    latest = df_all.iloc[i][lag_features].values.reshape(1, -1)
                    y_pred = model.predict(latest)[0]
                    df_all.at[i, "Water_level"] = y_pred
                    df_all.at[i, "Source"] = "Predicted"
                    preds.append((df_all.at[i, "Datetime"], y_pred))

            # ============================================
            # 7. Tampilkan tabel
            # ============================================
            st.success("✅ Data dan prediksi berhasil diproses.")
            st.dataframe(
                df_all[["Datetime", "Water_level", "Source", "Rainfall", "Cloud_cover",
                        "Surface_pressure", "Soil_temperature", "Soil_moisture"]]
                .tail(200)
                .style.format({"Water_level": "{:.3f}"})
            )

            # ============================================
            # 8. Plot hasil prediksi
            # ============================================
            fig = go.Figure()
            df_actual = df_all[df_all["Source"] == "Actual"]
            df_pred = df_all[df_all["Source"] == "Predicted"]

            fig.add_trace(go.Scatter(
                x=df_actual["Datetime"], y=df_actual["Water_level"],
                mode="lines+markers", name="Actual", line=dict(color="green", dash="dot")
            ))
            fig.add_trace(go.Scatter(
                x=df_pred["Datetime"], y=df_pred["Water_level"],
                mode="lines+markers", name="Predicted", line=dict(color="orange")
            ))

            fig.update_layout(
                title="Water Level Forecast (7-Day)",
                xaxis_title="Datetime (GMT+7)",
                yaxis_title="Water Level (m)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
