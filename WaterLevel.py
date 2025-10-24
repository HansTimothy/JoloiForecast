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

st.title("🌊 Water Level Forecast Dashboard")

# -----------------------------
# Current time (GMT+7), rounded up to next full hour
# -----------------------------
now_utc = datetime.utcnow()
gmt7_now = now_utc + timedelta(hours=7)
if gmt7_now.minute > 0 or gmt7_now.second > 0 or gmt7_now.microsecond > 0:
    rounded_now = (gmt7_now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
else:
    rounded_now = gmt7_now.replace(minute=0, second=0, microsecond=0)

# -----------------------------
# Select forecast start datetime
# -----------------------------
st.subheader("Select Start Date & Time for 7-Day Forecast")

selected_date = st.date_input(
    "Date",
    value=rounded_now.date(),
    max_value=rounded_now.date()
)

hour_options = [f"{h:02d}:00" for h in range(0, rounded_now.hour + 1)]
selected_hour_str = st.selectbox(
    "Hour",
    hour_options,
    index=len(hour_options)-1
)

selected_hour = int(selected_hour_str.split(":")[0])
start_datetime = datetime.combine(selected_date, time(selected_hour, 0, 0))
st.write(f"Start datetime (GMT+7): {start_datetime}")

# -----------------------------
# Upload water level data
# -----------------------------
st.subheader("Upload Hourly Water Level File")
uploaded_file = st.file_uploader("Upload CSV File (AWLR Joloi Logs)", type=["csv"])
wl_hourly = None

if uploaded_file is not None:
    try:
        df_wl = pd.read_csv(uploaded_file, engine='python', skip_blank_lines=True)
        if "Datetime" not in df_wl.columns or "Level Air" not in df_wl.columns:
            st.error("The file must contain columns 'Datetime' and 'Level Air'.")
        else:
            df_wl["Datetime"] = pd.to_datetime(df_wl["Datetime"]).dt.floor("H")
            start_limit = start_datetime - pd.Timedelta(hours=24)
            df_wl_filtered = df_wl[(df_wl["Datetime"] >= start_limit) & (df_wl["Datetime"] < start_datetime)]

            wl_hourly = (
                df_wl_filtered.groupby("Datetime")["Level Air"].mean().reset_index()
                .rename(columns={"Level Air": "Water_level"})
                .sort_values(by="Datetime", ascending=True)
                .round(2)
            )

            st.success("Successfully uploaded 24-hour water level data before start time.")
            st.dataframe(wl_hourly)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

# -----------------------------
# Fetch climate data
# -----------------------------
def fetch_climate_historical(start_dt, end_dt, lat=-0.117, lon=114.1):
    start_date = start_dt.date().isoformat()
    end_date = end_dt.date().isoformat()

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
            "Rainfall": data["hourly"]["rain"],
            "Cloud_cover": data["hourly"]["cloud_cover"],
            "Surface_pressure": data["hourly"]["surface_pressure"],
            "Soil_temperature": data["hourly"]["soil_temperature_0_to_7cm"],
            "Soil_moisture": data["hourly"]["soil_moisture_0_to_7cm"]
        })

        df["Datetime"] = df["Datetime"].dt.floor("H")
        return df

    except Exception as e:
        st.error(f"Failed to fetch historical climate data: {e}")
        return pd.DataFrame()

def fetch_climate_forecast(lat=-0.117, lon=114.1):
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
            "Rainfall": data["hourly"]["rain"],
            "Cloud_cover": data["hourly"]["cloud_cover"],
            "Surface_pressure": data["hourly"]["surface_pressure"],
            "Soil_temperature": data["hourly"]["soil_temperature_0cm"],
            "Soil_moisture": data["hourly"]["soil_moisture_0_to_1cm"]
        })

        df["Datetime"] = df["Datetime"].dt.floor("H")
        return df

    except Exception as e:
        st.error(f"Failed to fetch forecast climate data: {e}")
        return pd.DataFrame()

# -----------------------------
# Merge data & extend 7x24 hours
# -----------------------------
if wl_hourly is not None:
    if st.button("Fetch Climate Data"):
        start_dt = wl_hourly["Datetime"].min()
        end_dt = wl_hourly["Datetime"].max()
        climate_df = fetch_climate_historical(start_dt, end_dt)

        merged_df = (
            pd.merge(wl_hourly, climate_df, on="Datetime", how="left")
            .sort_values(by="Datetime", ascending=True)
        )

        # Generate next 7x24 hours
        next_hours = [start_datetime + timedelta(hours=i) for i in range(1, 168 + 1)]
        forecast_df = pd.DataFrame({"Datetime": next_hours})
        forecast_start, forecast_end = forecast_df["Datetime"].min(), forecast_df["Datetime"].max()

        # Determine source of climate data
        if forecast_end < gmt7_now:
            add_df = fetch_climate_historical(forecast_start, forecast_end)
        elif forecast_start > gmt7_now:
            add_df = fetch_climate_forecast()
        else:
            hist_df = fetch_climate_historical(forecast_start, gmt7_now)
            fore_df = fetch_climate_forecast()
            add_df = pd.concat([hist_df, fore_df]).drop_duplicates(subset="Datetime")

        forecast_merged = pd.merge(forecast_df, add_df, on="Datetime", how="left")
        forecast_merged["Water_level"] = np.nan
        forecast_merged["Source"] = "Forecast"

        merged_df["Source"] = "Historical"

        final_df = (
            pd.concat([merged_df, forecast_merged], ignore_index=True)
            .sort_values(by="Datetime", ascending=True)
        )

        # Round only numeric columns
        final_df = final_df.apply(lambda x: np.round(x, 2) if np.issubdtype(x.dtype, np.number) else x)

        # -----------------------------
        # Display Water Level + Climate Data
        # -----------------------------
        st.subheader("Water Level + Climate Data")
        
        # Ensure numeric columns are rounded and formatted to 2 decimals visually
        for col in final_df.select_dtypes(include=np.number).columns:
            final_df[col] = final_df[col].round(2)
        
        # Function for highlighting forecast rows
        def highlight_forecast(row):
            color = 'background-color: #cfe9ff' if row['Source'] == 'Forecast' else ''
            return [color] * len(row)
        
        # Format all numeric columns to 2 decimal places in the style
        format_dict = {col: "{:.2f}" for col in final_df.select_dtypes(include=np.number).columns}
        
        # Apply highlight + format
        styled_df = final_df.style.apply(highlight_forecast, axis=1).format(format_dict)
        
        # Display with auto-fit size
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )

# -----------------------------
# Rolling forecast 7x24 jam
# -----------------------------
if wl_hourly is not None and 'final_df' in locals():
    if st.button("Run Forecast"):
        st.subheader("Water Level 7-Day Forecast")

        # List kolom yang dipakai saat training
        highlighted_features = [
            "Rainfall_Lag17","Rainfall_Lag18","Rainfall_Lag19","Rainfall_Lag20",
            "Rainfall_Lag21","Rainfall_Lag22","Rainfall_Lag23","Rainfall_Lag24",
            "Cloud_cover_Lag1","Cloud_cover_Lag2","Cloud_cover_Lag3","Cloud_cover_Lag4","Cloud_cover_Lag5","Cloud_cover_Lag6","Cloud_cover_Lag7",
            "Surface_pressure_Lag1","Surface_pressure_Lag2","Surface_pressure_Lag3","Surface_pressure_Lag4","Surface_pressure_Lag5","Surface_pressure_Lag6",
            "Soil_temperature_Lag9","Soil_temperature_Lag10","Soil_temperature_Lag11",
            "Soil_moisture_Lag1","Soil_moisture_Lag2","Soil_moisture_Lag3","Soil_moisture_Lag4","Soil_moisture_Lag5","Soil_moisture_Lag6",
            "Water_level_Lag1","Water_level_Lag2","Water_level_Lag3","Water_level_Lag4","Water_level_Lag5"
        ]

        forecast_hours = 168  # 7 days x 24 hours
        forecast_results = []

        # Copy final_df supaya kita bisa update lag Water_level
        df_pred = final_df.copy()

        for i in range(forecast_hours):
            current_dt = start_datetime + timedelta(hours=i+1)

            # Ambil 24 jam terakhir + fitur lag yang dibutuhkan
            lag_data = {}
            for feat in highlighted_features:
                base, lag_num = feat.rsplit("_Lag", 1)
                lag_num = int(lag_num)
                lag_time = current_dt - timedelta(hours=lag_num)
                if base == "Water_level":
                    value = df_pred.loc[df_pred["Datetime"] == lag_time, "Water_level"].values
                else:
                    value = df_pred.loc[df_pred["Datetime"] == lag_time, base].values

                lag_data[feat] = value[0] if len(value) > 0 else np.nan

            X_pred = pd.DataFrame([lag_data])

            # Pastikan kolom urutannya sama persis seperti training
            X_pred = X_pred[highlighted_features]

            # Skip prediksi jika ada NaN (lag belum tersedia)
            if X_pred.isna().any().any():
                forecast_results.append({
                    "Datetime": current_dt,
                    "Water_level": np.nan,
                    "Source": "Forecast"
                })
                continue

            # Prediksi
            y_hat = model.predict(X_pred)[0]

            # Simpan hasil prediksi
            forecast_results.append({
                "Datetime": current_dt,
                "Water_level": y_hat,
                "Source": "Forecast"
            })

            # Update df_pred supaya lag Water_level berikutnya tersedia
            df_pred = pd.concat([df_pred, pd.DataFrame([{
                "Datetime": current_dt,
                "Water_level": y_hat
            }])], ignore_index=True)

        # Convert forecast_results ke DataFrame
        forecast_df_final = pd.DataFrame(forecast_results)

        # Tampilkan
        st.dataframe(
            forecast_df_final.style.format({"Water_level": "{:.2f}"}).apply(
                lambda row: ['background-color: #cfe9ff' if row['Source']=="Forecast" else '' for _ in row], axis=1
            ),
            use_container_width=True,
            height=400
        )
