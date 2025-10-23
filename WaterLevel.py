# WaterLevel_API_hourly_v2.py
import streamlit as st
import pandas as pd
import requests
import joblib
import pytz
import plotly.graph_objects as go
from datetime import datetime, timedelta, time

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("xgb_waterlevel_hourly_model.pkl")

st.title("Water Level Forecast Dashboard 🌊")

# waktu sekarang dibulatkan ke jam bawah
tz = pytz.timezone("Etc/GMT-7")  # GMT+7
now = datetime.now(tz)

if now.minute > 0 or now.second > 0 or now.microsecond > 0:
    rounded_now = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
else:
    rounded_now = now.replace(minute=0, second=0, microsecond=0)

# -----------------------------
# Pilih datetime start forecast
# -----------------------------
# Pilih tanggal & jam start forecast
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

# Ambil integer jam dari string
selected_hour = int(selected_hour_str.split(":")[0])

# Gabungkan menjadi datetime aware GMT+7
start_datetime = tz.localize(datetime.combine(selected_date, time(selected_hour, 0)))
st.write(f"Start datetime (GMT+7): {start_datetime}")

# -----------------------------
# Upload water level file
# -----------------------------
st.subheader("Upload Water Level File (Hourly)")
uploaded_file = st.file_uploader("Upload file CSV AWLR Logs Joloi", type=["csv"])

wl_hourly = None
if uploaded_file is not None:
    try:
        # baca CSV
        df_wl = pd.read_csv(uploaded_file, engine='python', skip_blank_lines=True)
        if "Datetime" not in df_wl.columns or "Level Air" not in df_wl.columns:
            st.error("File harus memiliki kolom 'Datetime' dan 'Level Air'")
        else:
            # ubah ke datetime GMT+7
            df_wl["Datetime"] = pd.to_datetime(df_wl["Datetime"]).dt.tz_localize(tz)
            
            # filter 24 jam sebelum start_datetime
            start_limit = start_datetime - timedelta(hours=24)
            df_wl = df_wl[(df_wl["Datetime"] >= start_limit) & (df_wl["Datetime"] < start_datetime)]
            
            # group by jam
            df_wl["Hour"] = df_wl["Datetime"].dt.floor("H")
            wl_hourly = df_wl.groupby("Hour")["Level Air"].mean().reset_index()
            wl_hourly.rename(columns={"Hour": "Datetime", "Level Air": "Water_level"}, inplace=True)
            
            st.success(f"Data water level berhasil diupload)")
            st.dataframe(wl_hourly.style.format({"Water_level":"{:.2f}"}))
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")

# -----------------------------
# Fetch data & predict
# -----------------------------
if st.button("Fetch Weather & Predict") and wl_hourly is not None:
    # -----------------------------
    # Ambil Historical (archive) + Forecast Open-Meteo
    # -----------------------------
    # Historical: 7 hari sebelum start_datetime
    start_hist = (start_datetime - timedelta(days=7)).date()
    end_hist = (start_datetime - timedelta(hours=1)).date()
    
    url_hist = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude=-0.117&longitude=114.1&start_date={start_hist}&end_date={end_hist}"
        f"&hourly=rain,cloud_cover,surface_pressure,soil_moisture_0_to_1cm,soil_temperature_0cm"
        f"&timezone=Asia%2FSingapore"
    )
    hist = requests.get(url_hist).json()
    
    df_hist = pd.DataFrame(hist["hourly"])
    df_hist["time"] = pd.to_datetime(df_hist["time"])
    df_hist.set_index("time", inplace=True)
    
    # Forecast: 7 hari ke depan
    url_forecast = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude=-0.117&longitude=114.1"
        f"&hourly=rain,cloud_cover,surface_pressure,soil_moisture_0_to_1cm,soil_temperature_0cm"
        f"&timezone=Asia%2FSingapore&forecast_days=7"
    )
    forecast = requests.get(url_forecast).json()
    df_forecast = pd.DataFrame(forecast["hourly"])
    df_forecast["time"] = pd.to_datetime(df_forecast["time"])
    df_forecast.set_index("time", inplace=True)
    
    # Gabungkan hist + forecast
    df_weather = pd.concat([df_hist, df_forecast])
    df_weather = df_weather.sort_index()
    
    # -----------------------------
    # Siapkan dataframe prediksi hourly
    # -----------------------------
    forecast_hours = [start_datetime + timedelta(hours=i) for i in range(7*24)]
    df_pred = pd.DataFrame({"Datetime": forecast_hours})
    df_pred["Date"] = df_pred["Datetime"].dt.date
    df_pred = df_pred.merge(df_weather.reset_index(), left_on="Datetime", right_on="time", how="left")
    df_pred.drop(columns=["time","Date"], inplace=True)
    
    # Masukkan water level historis dari upload
    wl_dict = dict(zip(wl_hourly["Datetime"], wl_hourly["Water_level"]))
    df_pred["Water_level"] = df_pred["Datetime"].map(wl_dict)
    
    df_pred.set_index("Datetime", inplace=True)
    
    # -----------------------------
    # Buat lag features sesuai daftar
    # -----------------------------
    lag_features = []
    # Rainfall lag 17-24
    for i in range(17,25):
        lag_features.append(f"Rainfall_Lag{i}")
    # Cloud cover lag 1-24
    for i in range(1,25):
        lag_features.append(f"Cloud_cover_Lag{i}")
    # Surface pressure lag 1-24
    for i in range(1,25):
        lag_features.append(f"Surface_pressure_Lag{i}")
    # Soil temperature lag 9-11
    for i in range(9,12):
        lag_features.append(f"Soil_temperature_Lag{i}")
    # Soil moisture lag 1-24
    for i in range(1,25):
        lag_features.append(f"Soil_moisture_Lag{i}")
    # Water level lag 1-24
    for i in range(1,25):
        lag_features.append(f"Water_level_Lag{i}")
    
    features = lag_features
    results = {}
    
    # -----------------------------
    # Helper untuk ambil nilai aman
    # -----------------------------
    def safe_get(df, dt, col):
        try:
            return float(df.loc[dt, col])
        except:
            return 0.0
    
    # -----------------------------
    # Rolling lags
    # -----------------------------
    water_level_lags = [safe_get(df_pred, dt, "Water_level") for dt in wl_hourly["Datetime"][-24:]]
    
    for dt in df_pred.index:
        inp = {}
        # Rainfall lags 17-24
        for i in range(17,25):
            inp[f"Rainfall_Lag{i}"] = [safe_get(df_pred, dt - timedelta(hours=i), "rain")]
        # Cloud cover lags 1-24
        for i in range(1,25):
            inp[f"Cloud_cover_Lag{i}"] = [safe_get(df_pred, dt - timedelta(hours=i), "cloud_cover")]
        # Surface pressure lags 1-24
        for i in range(1,25):
            inp[f"Surface_pressure_Lag{i}"] = [safe_get(df_pred, dt - timedelta(hours=i), "surface_pressure")]
        # Soil temperature lag 9-11
        for i in range(9,12):
            inp[f"Soil_temperature_Lag{i}"] = [safe_get(df_pred, dt - timedelta(hours=i), "soil_temperature_0cm")]
        # Soil moisture lag 1-24
        for i in range(1,25):
            inp[f"Soil_moisture_Lag{i}"] = [safe_get(df_pred, dt - timedelta(hours=i), "soil_moisture_0_to_1cm")]
        # Water level lag 1-24
        for i in range(1,25):
            inp[f"Water_level_Lag{i}"] = [water_level_lags[i-1]]
        
        input_data = pd.DataFrame(inp)[features].fillna(0.0)
        pred = model.predict(input_data)[0]
        df_pred.loc[dt,"Water_level"] = round(pred,2)
        results[dt] = pred
        water_level_lags = [pred] + water_level_lags[:-1]
    
    st.subheader("Forecast Results (Hourly)")
    st.dataframe(df_pred[["Water_level"]].style.format("{:.2f}"))
    
    # -----------------------------
    # Plot
    # -----------------------------
    fig = go.Figure()
    df_plot = df_pred.reset_index()
    fig.add_trace(go.Scatter(
        x=df_plot["Datetime"],
        y=df_plot["Water_level"],
        mode="lines+markers",
        line=dict(color="blue", width=2),
        marker=dict(size=4),
        name="Predicted Water Level"
    ))
    st.plotly_chart(fig, use_container_width=True)
