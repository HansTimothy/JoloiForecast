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
    st.info(f"Fetching climate data...")

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
            "Surface_pressure": data["hourly"]["surface_pressure"],
            "Cloud_cover": data["hourly"]["cloud_cover"],
            "Soil_temperature": data["hourly"]["soil_temperature_0_to_7cm"],
            "Soil_moisture": data["hourly"]["soil_moisture_0_to_7cm"],
            "Rainfall": data["hourly"]["rain"]
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
            "Surface_pressure": data["hourly"]["surface_pressure"],
            "Cloud_cover": data["hourly"]["cloud_cover"],
            "Soil_temperature": data["hourly"]["soil_temperature_0cm"],
            "Soil_moisture": data["hourly"]["soil_moisture_0_to_1cm"],
            "Rainfall": data["hourly"]["rain"]
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

        merged_df["Source"] = "Observed"

        final_df = (
            pd.concat([merged_df, forecast_merged], ignore_index=True)
            .sort_values(by="Datetime", ascending=True)
        )

        # Round only numeric columns
        final_df = final_df.apply(lambda x: np.round(x, 2) if np.issubdtype(x.dtype, np.number) else x)

        # -----------------------------
        # Highlight forecast rows (blue) but hide "Source" column
        # -----------------------------
        st.subheader("Water Level + Climate Data")
        
        # Columns to display (hide 'Source')
        display_cols = [col for col in final_df.columns if col != "Source"]
        
        # Function to highlight forecast rows
        def highlight_blue(row):
            color = "background-color: lightblue" if row["Source"] == "Forecast" else ""
            return [color] * len(row)
        
        # Apply rounding to 2 decimals on numeric columns
        for col in final_df.select_dtypes(include=np.number).columns:
            final_df[col] = final_df[col].round(2)
        
        # Apply highlight and formatting
        styled_df = (
            final_df.style
            .apply(highlight_blue, axis=1)
            .format(precision=2)  # display numbers with 2 decimals
            .hide(axis="columns", subset=["Source"])  # hide the 'Source' column
        )
        
        # Render styled dataframe with HTML (to keep color)
        st.markdown(styled_df.to_html(escape=False), unsafe_allow_html=True)
