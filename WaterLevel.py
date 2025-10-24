import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np
from datetime import datetime, timedelta, time
from xgboost import XGBRegressor
import time as t
from io import BytesIO
from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import plotly.graph_objects as go

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("xgb_waterlevel_hourly_model.pkl")

st.title("🌊 Water Level Forecast Dashboard")

# -----------------------------
# Initialize session state
# -----------------------------
if "forecast_done" not in st.session_state:
    st.session_state["forecast_done"] = False
if "selected_datetime" not in st.session_state:
    st.session_state["selected_datetime"] = None
if "last_uploaded_name" not in st.session_state:
    st.session_state["last_uploaded_name"] = None

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
selected_date = st.date_input("Date", value=rounded_now.date(), max_value=rounded_now.date())
hour_options = [f"{h:02d}:00" for h in range(0, rounded_now.hour + 1)]
selected_hour_str = st.selectbox("Hour", hour_options, index=len(hour_options) - 1)
selected_hour = int(selected_hour_str.split(":")[0])
start_datetime = datetime.combine(selected_date, time(selected_hour, 0, 0))
st.write(f"Start datetime (GMT+7): {start_datetime}")

# Reset forecast if date/time changed
if st.session_state["selected_datetime"] != start_datetime:
    st.session_state["selected_datetime"] = start_datetime
    st.session_state["forecast_done"] = False

# -----------------------------
# Instructions
# -----------------------------
st.subheader("Instructions for Uploading Water Level Data")
st.info(
    f"Please upload a CSV file containing hourly water level data.\n"
    f"- The CSV must have columns: 'Datetime' and 'Level Air'.\n"
    f"- 'Datetime' should be in proper datetime format (e.g., YYYY-MM-DD HH:MM:SS).\n"
    f"- The data should cover **the last 24 hours before the selected start datetime** "
    f"({start_datetime - timedelta(hours=24)} to {start_datetime}).\n"
    f"- Make sure there are no missing hours in this period."
)

# -----------------------------
# Upload water level data
# -----------------------------
st.subheader("Upload Hourly Water Level File")
uploaded_file = st.file_uploader("Upload CSV File (AWLR Joloi Logs)", type=["csv"])
wl_hourly = None
upload_success = False

# Reset forecast if new file uploaded
if uploaded_file is not None:
    if st.session_state["last_uploaded_name"] != uploaded_file.name:
        st.session_state["last_uploaded_name"] = uploaded_file.name
        st.session_state["forecast_done"] = False

if uploaded_file is not None:
    try:
        df_wl = pd.read_csv(uploaded_file, engine="python", skip_blank_lines=True)
        if "Datetime" not in df_wl.columns or "Level Air" not in df_wl.columns:
            st.error("The file must contain columns 'Datetime' and 'Level Air'.")
        else:
            df_wl["Datetime"] = pd.to_datetime(df_wl["Datetime"]).dt.floor("H")
            start_limit = start_datetime - pd.Timedelta(hours=24)
            df_wl_filtered = df_wl[
                (df_wl["Datetime"] >= start_limit) & (df_wl["Datetime"] < start_datetime)
            ]

            # Check completeness
            expected_hours = pd.date_range(
                start=start_limit, end=start_datetime - pd.Timedelta(hours=1), freq="H"
            )
            actual_hours = pd.to_datetime(df_wl_filtered["Datetime"].sort_values().unique())
            missing_hours = sorted(set(expected_hours) - set(actual_hours))
            if missing_hours:
                missing_str = ", ".join([dt.strftime("%Y-%m-%d %H:%M") for dt in missing_hours])
                st.warning(f"The uploaded water level data is incomplete! Missing hours: {missing_str}")
            else:
                upload_success = True
                wl_hourly = (
                    df_wl_filtered.groupby("Datetime")["Level Air"]
                    .mean()
                    .reset_index()
                    .rename(columns={"Level Air": "Water_level"})
                    .sort_values(by="Datetime", ascending=True)
                    .round(2)
                )
                st.success("✅ File successfully uploaded and verified.")
                st.dataframe(wl_hourly)

    except Exception as e:
        st.error(f"Failed to read file: {e}")

# -----------------------------
# Fetch climate functions
# -----------------------------
def fetch_climate_historical(start_dt, end_dt, lat=-0.117, lon=114.1):
    try:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_dt.date()}&end_date={end_dt.date()}"
            f"&hourly=surface_pressure,cloud_cover,soil_temperature_0_to_7cm,soil_moisture_0_to_7cm,rain"
            f"&timezone=Asia%2FBangkok"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame({
            "Datetime": pd.to_datetime(data["hourly"]["time"]),
            "Rainfall": data["hourly"]["rain"],
            "Cloud_cover": data["hourly"]["cloud_cover"],
            "Surface_pressure": data["hourly"]["surface_pressure"],
            "Soil_temperature": data["hourly"]["soil_temperature_0_to_7cm"],
            "Soil_moisture": data["hourly"]["soil_moisture_0_to_7cm"],
        })
        df["Datetime"] = df["Datetime"].dt.floor("H")
        return df
    except Exception as e:
        st.error(f"Failed to fetch climate data: {e}")
        return pd.DataFrame()


def fetch_climate_forecast(lat=-0.117, lon=114.1):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&hourly=rain,surface_pressure,cloud_cover,soil_moisture_0_to_1cm,soil_temperature_0cm"
            f"&timezone=Asia%2FBangkok&forecast_days=14"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame({
            "Datetime": pd.to_datetime(data["hourly"]["time"]),
            "Rainfall": data["hourly"]["rain"],
            "Cloud_cover": data["hourly"]["cloud_cover"],
            "Surface_pressure": data["hourly"]["surface_pressure"],
            "Soil_temperature": data["hourly"]["soil_temperature_0cm"],
            "Soil_moisture": data["hourly"]["soil_moisture_0_to_1cm"],
        })
        df["Datetime"] = df["Datetime"].dt.floor("H")
        return df
    except Exception as e:
        st.error(f"Failed to fetch climate data: {e}")
        return pd.DataFrame()

# -----------------------------
# Run 7-Day Forecast
# -----------------------------
run_forecast = st.button("Run 7-Day Forecast")

if upload_success and run_forecast:
    st.session_state["forecast_done"] = False

    progress_container = st.empty()
    progress_bar = st.progress(0)
    total_steps = 3 + 168
    step_counter = 0

    # 1️⃣ Fetch climate
    progress_container.markdown("Fetching climate data...")
    start_dt = wl_hourly["Datetime"].min()
    end_dt = wl_hourly["Datetime"].max()
    climate_hist = fetch_climate_historical(start_dt, end_dt)
    forecast_hours = [start_datetime + timedelta(hours=i) for i in range(0, 168)]
    forecast_df = pd.DataFrame({"Datetime": forecast_hours})

    hist_df = fetch_climate_historical(forecast_df["Datetime"].min(), datetime.now() + timedelta(hours=7))
    fore_df = fetch_climate_forecast()
    climate_forecast = pd.concat([hist_df, fore_df]).drop_duplicates(subset="Datetime")

    step_counter += 1
    progress_bar.progress(step_counter / total_steps)

    # 2️⃣ Merge data
    progress_container.markdown("Merging water level and climate data...")
    merged_df = pd.merge(wl_hourly, climate_hist, on="Datetime", how="left").sort_values("Datetime")
    merged_df["Source"] = "Historical"
    forecast_merged = pd.merge(forecast_df, climate_forecast, on="Datetime", how="left")
    forecast_merged["Water_level"] = np.nan
    forecast_merged["Source"] = "Forecast"
    final_df = pd.concat([merged_df, forecast_merged], ignore_index=True).sort_values("Datetime")
    final_df = final_df.apply(lambda x: np.round(x, 2) if np.issubdtype(x.dtype, np.number) else x)
    step_counter += 1
    progress_bar.progress(step_counter / total_steps)

    # 3️⃣ Forecast iteratively
    progress_container.markdown("Forecasting water level for 7 days...")
    model_features = model.get_booster().feature_names
    forecast_indices = final_df.index[final_df["Source"] == "Forecast"]

    for i, idx in enumerate(forecast_indices, start=1):
        step_counter += 1
        progress_bar.progress(step_counter / total_steps)
        progress_container.markdown(f"Forecasting hour {i}/{len(forecast_indices)}...")

        X_forecast = pd.DataFrame(columns=model_features, index=[0])
        for f in model_features:
            base, lag = f.rsplit("_Lag", 1)
            lag = int(lag)
            try:
                X_forecast.at[0, f] = final_df.loc[idx - lag, base]
            except:
                X_forecast.at[0, f] = final_df.loc[final_df["Source"] == "Historical", base].iloc[-lag]
        X_forecast = X_forecast.astype(float)
        y_hat = model.predict(X_forecast)[0]
        final_df.at[idx, "Water_level"] = max(0.0, round(y_hat, 2))

    progress_container.markdown("✅ 7-Day Water Level Forecast Completed!")
    progress_bar.progress(1.0)

    st.session_state["final_df"] = final_df
    st.session_state["forecast_done"] = True

# -----------------------------
# Show results if already done
# -----------------------------
if st.session_state.get("forecast_done", False):
    final_df = st.session_state["final_df"]

    st.subheader("Water Level + Climate Data with Forecast")
    def highlight_forecast(row):
        return ['background-color: #cfe9ff' if row["Source"] == "Forecast" else '' for _ in row]
    st.dataframe(final_df.style.apply(highlight_forecast, axis=1), use_container_width=True, height=500)

    # Plot
    st.subheader("Water Level Forecast Plot")
    rmse_est = 0.2
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=final_df.query("Source=='Historical'")["Datetime"],
        y=final_df.query("Source=='Historical'")["Water_level"],
        mode="lines+markers", name="Historical", line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=final_df.query("Source=='Forecast'")["Datetime"],
        y=final_df.query("Source=='Forecast'")["Water_level"],
        mode="lines+markers", name="Forecast", line=dict(color="orange")
    ))
    forecast_x = final_df.query("Source=='Forecast'")["Datetime"]
    forecast_y = final_df.query("Source=='Forecast'")["Water_level"]
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_x, forecast_x[::-1]]),
        y=pd.concat([forecast_y + rmse_est, (forecast_y - rmse_est).clip(0)[::-1]]),
        fill="toself", fillcolor="rgba(255,165,0,0.2)", line=dict(color="rgba(255,255,255,0)"),
        name="Forecast ± RMSE"
    ))
    fig.update_layout(title="Water Level Historical vs 7-Day Forecast", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Download buttons (centered)
    # -----------------------------
    export_df = final_df[["Datetime", "Water_level"]].copy()
    export_df["Datetime"] = export_df["Datetime"].astype(str)

    csv_data = export_df.to_csv(index=False).encode("utf-8")
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Forecast")
    excel_buffer.seek(0)

    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    data = [export_df.columns.tolist()] + export_df.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#007acc")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ]))
    doc.build([Paragraph("Joloi Water Level Forecast", styles["Title"]), table])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("Download CSV", csv_data, "water_level_forecast.csv", "text/csv", use_container_width=True)
    with col2:
        st.download_button("Download Excel", excel_buffer, "water_level_forecast.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with col3:
        st.download_button("Download PDF", pdf_buffer.getvalue(), "water_level_forecast.pdf", "application/pdf", use_container_width=True)
