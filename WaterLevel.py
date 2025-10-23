# WaterLevel_API_fixed.py
import streamlit as st
import pandas as pd
import requests
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("xgb_waterlevel_hourly_model.pkl")

st.title("Water Level Prediction Dashboard 🌊")

today = datetime.today().date()

st.subheader("Pilih Tanggal Prediksi Water Level (Max: H+14)")

pred_date = st.date_input(
    "Tanggal Prediksi Water Level",
    value=today + timedelta(days=1),
    min_value=date(2018, 5, 8),
    max_value=today + timedelta(days=14)
)

# features (dipakai di kedua skenario)
features = [
    "Precipitation_lag1d","Precipitation_lag2d","Precipitation_lag3d","Precipitation_lag4d",
    "Precipitation_lag5d","Precipitation_lag6d","Precipitation_lag7d",
    "Temperature_lag1d","Temperature_lag2d","Temperature_lag3d","Temperature_lag4d",
    "Relative_humidity_lag1d","Relative_humidity_lag2d","Relative_humidity_lag3d",
    "Relative_humidity_lag4d","Relative_humidity_lag5d","Relative_humidity_lag6d",
    "Relative_humidity_lag7d",
    "Water_level_lag1d","Water_level_lag2d","Water_level_lag3d","Water_level_lag4d",
    "Water_level_lag5d","Water_level_lag6d","Water_level_lag7d",
]

# -----------------------------
# Tentukan tanggal untuk input Water Level manual
# -----------------------------
st.subheader("Masukkan Data Water Level")

if pred_date <= today + timedelta(days=1):
    # Prediksi untuk masa lalu atau H+1:
    wl_dates = [pred_date - timedelta(days=i) for i in range(1, 8)]
else:
    # Prediksi H+2..H+7:
    wl_dates = [today - timedelta(days=i) for i in range(0, 7)]

wl_inputs = [
    st.number_input(f"Water Level **{d.strftime('%d %B %Y')}**", value=20.0, format="%.2f")
    for d in wl_dates
]

# -----------------------------
# Helper untuk ambil value dari df dengan fallback
# -----------------------------
def safe_get(df, date_key, col):
    try:
        return float(df.loc[date_key, col])
    except Exception:
        return 0.0

# -----------------------------
# Fetch data & predict
# -----------------------------
if st.button("Fetch Data & Predict"):

    if pred_date <= today + timedelta(days=1):
        # -----------------------------
        # Skenario 1: prediksi masa lalu / H+1 (pakai archive)
        # -----------------------------
        start_date = (pred_date - timedelta(days=7)).isoformat()
        end_date = (pred_date - timedelta(days=1)).isoformat()

        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude=-0.61&longitude=114.8&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
            f"&timezone=Asia%2FSingapore"
        )
        data = requests.get(url).json()

        df = pd.DataFrame({
            "time": data["daily"]["time"],
            "precipitation_sum": data["daily"]["precipitation_sum"],
            "temperature_mean": data["daily"]["temperature_2m_mean"],
            "relative_humidity": data["daily"]["relative_humidity_2m_mean"]
        })
        df["time"] = pd.to_datetime(df["time"]).dt.date
        df.set_index("time", inplace=True)

        # Tambah kolom water level manual (dari input wl_inputs)
        for i, d in enumerate(wl_dates):
            if d in df.index:
                df.loc[d, "water_level"] = round(float(wl_inputs[i]), 2)

        df_preview = df.copy()
        if "water_level" in df_preview.columns:
            wl = df_preview.pop("water_level")
            df_preview.insert(0, "water_level", wl)
        
        numeric_cols = ["precipitation_sum", "temperature_mean", "water_level"]
        st.subheader("Preview Data Historis")
        st.dataframe(df_preview.style.format("{:.2f}", subset=numeric_cols).set_properties(**{"text-align": "right"}, subset=numeric_cols))

        # buat input feature untuk pred_date
        inp = {}
        for i in range(1, 8):
            date_i = pred_date - timedelta(days=i)
            inp[f"Precipitation_lag{i}d"] = [safe_get(df, date_i, "precipitation_sum")]
        for i in range(1, 5):
            date_i = pred_date - timedelta(days=i)
            inp[f"Temperature_lag{i}d"] = [safe_get(df, date_i, "temperature_mean")]
        for i in range(1, 8):
            date_i = pred_date - timedelta(days=i)
            inp[f"Relative_humidity_lag{i}d"] = [safe_get(df, date_i, "relative_humidity")]
        for i in range(1, 8):
            # wl_inputs[0] == pred_date-1, wl_inputs[6] == pred_date-7
            inp[f"Water_level_lag{i}d"] = [wl_inputs[i-1]]

        input_data = pd.DataFrame(inp)[features].fillna(0.0)

        prediction = model.predict(input_data)[0]
        df.loc[pred_date, "water_level"] = round(prediction, 2)
        st.success(f"Predicted Water Level on {pred_date.strftime('%d %B %Y')}: {prediction:.2f} m")

    else:
        # -----------------------------
        # Skenario 2: prediksi H+2..H+14 (pakai archive + forecast)
        # -----------------------------
        start_hist = (today - timedelta(days=6)).isoformat()
        end_hist = today.isoformat()
        url_hist = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude=-0.61&longitude=114.8&start_date={start_hist}&end_date={end_hist}"
            f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
            f"&timezone=Asia%2FSingapore"
        )
        hist = requests.get(url_hist).json()
        df_hist = pd.DataFrame({
            "time": hist["daily"]["time"],
            "precipitation_sum": hist["daily"]["precipitation_sum"],
            "temperature_mean": hist["daily"]["temperature_2m_mean"],
            "relative_humidity": hist["daily"]["relative_humidity_2m_mean"]
        })
        df_hist["time"] = pd.to_datetime(df_hist["time"]).dt.date
        
        n_days = (pred_date - today).days  # misal H+2 -> n_days=2
        
        url_forecast = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude=-0.61&longitude=114.8"
            f"&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
            f"&timezone=Asia%2FSingapore&forecast_days=16"
        )
        forecast = requests.get(url_forecast).json()
        df_forecast = pd.DataFrame({
            "time": forecast["daily"]["time"],
            "precipitation_sum": forecast["daily"]["precipitation_sum"],
            "temperature_mean": forecast["daily"]["temperature_2m_mean"],
            "relative_humidity": forecast["daily"]["relative_humidity_2m_mean"]
        })
        # ambil H+1 .. H+n_days
        df_forecast = df_forecast.iloc[1:n_days+1]
        df_forecast["time"] = pd.to_datetime(df_forecast["time"]).dt.date
        
        # gabungkan histori + forecast
        df = pd.concat([df_hist, df_forecast]).drop_duplicates().sort_values("time")
        df.set_index("time", inplace=True)
        df["water_level"] = None
        
        # Isi dari input manual (H0..H-6)
        for i, d in enumerate(wl_dates):
            if d in df.index:
                df.loc[d, "water_level"] = round(float(wl_inputs[i]), 2)
        
        # Siapkan lags untuk water level (H0 terbaru -> index 0)
        water_level_lags = wl_inputs[:]  # [H0, H-1, H-2, ..., H-6]
        
        results = {}
        forecast_dates = []  # untuk highlight
        
        for step in range(1, n_days + 1):
            pred_day = today + timedelta(days=step)
        
            inp = {}
            # lags cuaca
            for i in range(1, 8):
                date_i = pred_day - timedelta(days=i)
                inp[f"Precipitation_lag{i}d"] = [safe_get(df, date_i, "precipitation_sum")]
            for i in range(1, 5):
                date_i = pred_day - timedelta(days=i)
                inp[f"Temperature_lag{i}d"] = [safe_get(df, date_i, "temperature_mean")]
            for i in range(1, 8):
                date_i = pred_day - timedelta(days=i)
                inp[f"Relative_humidity_lag{i}d"] = [safe_get(df, date_i, "relative_humidity")]
        
            # lags water level
            for i in range(1, 8):
                inp[f"Water_level_lag{i}d"] = [water_level_lags[i-1]]
        
            # buat dataframe input dan prediksi
            input_data = pd.DataFrame(inp)[features].fillna(0.0)
            prediction = model.predict(input_data)[0]
            results[pred_day] = prediction
        
            # update lags
            water_level_lags = [prediction] + water_level_lags[:-1]
        
            # update df (biar preview langsung terlihat)
            if pred_day in df.index:
                df.loc[pred_day, "water_level"] = round(prediction, 2)
                forecast_dates.append(pred_day)
        
        # --- letakkan water_level di kolom pertama untuk preview ---
        df_preview = df.copy()
        if "water_level" in df_preview.columns:
            wl = df_preview.pop("water_level")
            df_preview.insert(0, "water_level", wl)
        
        df_preview = df_preview.iloc[:-1]
        
        # styling untuk highlight forecast
        def highlight_forecast(row):
            return ['background-color: #bce4f6' if row.name in forecast_dates else '' for _ in row]
        
        numeric_cols = ["precipitation_sum", "temperature_mean", "water_level"]
        
        st.subheader("Preview Data (History + Forecast)")
        st.dataframe(df_preview.style.apply(highlight_forecast, axis=1).format("{:.2f}", subset=numeric_cols).set_properties(**{"text-align": "right"}, subset=numeric_cols))
        
        # Ambil hasil prediksi terakhir
        last_date, last_val = list(results.items())[-1]
        st.subheader("Hasil Prediksi")
        st.success(f"Predicted Water Level on {last_date.strftime('%d %B %Y')}: {last_val:.2f} m")

    # -----------------------------
    # PLOT SECTION (fixed for dashed continuity + RMSE band)
    # -----------------------------
    df_plot = df.reset_index()[["time", "water_level"]].copy()
    df_plot.rename(columns={"time": "Date"}, inplace=True)

    # ensure numeric water_level
    df_plot["water_level"] = pd.to_numeric(df_plot["water_level"], errors="coerce")

    lower_limit = 19.5
    upper_limit = 28.5
    today = datetime.today().date()

    # --- Tentukan rentang histori & prediksi ---
    if pred_date <= today + timedelta(days=1):
        # prediksi historis (pred_date termasuk ke df_pred)
        df_hist = df_plot[df_plot["Date"] < pred_date]
        df_pred = df_plot[df_plot["Date"] >= pred_date]
        # continuity start = pred_date - 1
        continuity_start = pred_date - timedelta(days=1)
    else:
        # prediksi masa depan: histori sampai today, prediksi mulai hari setelah today
        df_hist = df_plot[df_plot["Date"] <= today]
        df_pred = df_plot[df_plot["Date"] > today]
        continuity_start = today

    # split prediksi aman / tidak aman (ignore NaN)
    df_pred_safe = df_pred[df_pred["water_level"].between(lower_limit, upper_limit)]
    df_pred_unsafe = df_pred[(df_pred["water_level"] < lower_limit) | (df_pred["water_level"] > upper_limit)]

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=df_hist["Date"],
        y=df_hist["water_level"],
        mode="lines",  # hanya garis, tanpa legend
        line=dict(color="blue", width=2),
        showlegend=False,  # ✅ tidak muncul di legend
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=df_hist["Date"],
        y=df_hist["water_level"],
        mode="markers",
        marker=dict(color="blue", size=8),
        showlegend=True,
        name="Historical"
    ))
    
    # --- Dashed continuity line: konekkan titik historis terakhir -> titik prediksi pertama ---
    # cari nilai historis di continuity_start (jika ada)
    last_hist_val = None
    last_hist_row = df_plot[df_plot["Date"] == continuity_start]
    if not last_hist_row.empty and pd.notna(last_hist_row["water_level"].iloc[0]):
        last_hist_val = float(last_hist_row["water_level"].iloc[0])

    # ambil hanya prediksi yang memiliki nilai numeric
    df_pred_nonnull = df_pred.dropna(subset=["water_level"]).copy()
    pred_dates_numeric = df_pred_nonnull["Date"].tolist()
    pred_values_numeric = df_pred_nonnull["water_level"].tolist()

    # prepare dashed line coordinates
    dashed_x = []
    dashed_y = []
    if last_hist_val is not None and len(pred_dates_numeric) >= 1:
        dashed_x = [continuity_start] + pred_dates_numeric
        dashed_y = [last_hist_val] + pred_values_numeric
    elif len(pred_dates_numeric) >= 2:
        dashed_x = pred_dates_numeric
        dashed_y = pred_values_numeric
    # add dashed line if enough points
    if len(dashed_x) >= 2:
        fig.add_trace(go.Scatter(
            x=dashed_x,
            y=dashed_y,
            mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip"
        ))

    # Plot prediksi (Loadable & Unloadable) di atas dashed line
    fig.add_trace(go.Scatter(
        x=df_pred_safe["Date"],
        y=df_pred_safe["water_level"],
        mode="markers",
        marker=dict(color="green", size=8),
        name="Prediction (Loadable)"
    ))
    fig.add_trace(go.Scatter(
        x=df_pred_unsafe["Date"],
        y=df_pred_unsafe["water_level"],
        mode="markers",
        marker=dict(color="red", size=8),
        name="Prediction (Unloadable)"
    ))

    # Today marker (jika ada)
    today_point = df_plot[df_plot["Date"] == today]
    if not today_point.empty and pd.notna(today_point["water_level"].iloc[0]):
        fig.add_trace(go.Scatter(
            x=today_point["Date"],
            y=today_point["water_level"],
            mode="markers",
            marker=dict(color="blue", size=8, symbol="circle"),
            name="Today",
            showlegend=False
        ))

    # Loadable limits
    fig.add_hline(y=lower_limit, line=dict(color="red", width=2, dash="dash"),
                  annotation_text="Lower Limit", annotation_position="bottom left")
    fig.add_hline(y=upper_limit, line=dict(color="red", width=2, dash="dash"),
                  annotation_text="Upper Limit", annotation_position="top left")

    # --- RMSE area: gunakan continuity_start sebagai titik awal (jika ada), fallback ke pred-only if >=2 points ---
    rmse = 0.87
    band_added = False
    if last_hist_val is not None and len(pred_dates_numeric) >= 1:
        band_x = [continuity_start] + pred_dates_numeric
        upper_band = [last_hist_val + rmse] + [v + rmse for v in pred_values_numeric]
        lower_band = [last_hist_val - rmse] + [v - rmse for v in pred_values_numeric]
        band_added = True
    elif len(pred_values_numeric) >= 2:
        band_x = pred_dates_numeric
        upper_band = [v + rmse for v in pred_values_numeric]
        lower_band = [v - rmse for v in pred_values_numeric]
        band_added = True

    if band_added:
        # lower band trace (invisible line, used as baseline for fill)
        fig.add_trace(go.Scatter(
            x=band_x,
            y=lower_band,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        # upper band with fill to previous trace (tonexty)
        fig.add_trace(go.Scatter(
            x=band_x,
            y=upper_band,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)", dash="dash"),
            fill="tonexty",
            fillcolor="rgba(0, 0, 255, 0.12)",
            name=f"Prediction error ±{rmse:.2f} m",
            showlegend=True
        ))

    # Format tanggal tick
    all_dates = df_plot["Date"]
    tick_text = [d.strftime("%d/%m/%y") for d in all_dates]

    fig.update_layout(
        title="Water Level Dashboard 🌊",
        xaxis_title="Date",
        yaxis_title="Water Level (m)",
        xaxis=dict(
            tickangle=90,
            tickmode="array",
            tickvals=all_dates,
            ticktext=tick_text
        ),
        yaxis=dict(autorange=True),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
