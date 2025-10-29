import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import datetime, timedelta, time
from io import BytesIO
from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import plotly.graph_objects as go

# -----------------------------
# Load trained XGB model
# -----------------------------
model = joblib.load("xgb_waterlevel_hourly_model.pkl")
st.title("🌊 Water Level Forecast Dashboard")

# -----------------------------
# Current time (GMT+7), rounded up to next full hour
# -----------------------------
now_utc = datetime.utcnow()
gmt7_now = now_utc + timedelta(hours=7)
rounded_now = gmt7_now.replace(minute=0, second=0, microsecond=0)
if gmt7_now.minute > 0 or gmt7_now.second > 0:
    rounded_now += timedelta(hours=1)

# -----------------------------
# Select forecast start datetime
# -----------------------------
st.subheader("Select Start Date & Time for 7-Day Forecast")
selected_date = st.date_input("Date", value=rounded_now.date(), max_value=rounded_now.date())
hour_options = [f"{h:02d}:00" for h in range(0, rounded_now.hour + 1)] if selected_date == rounded_now.date() else [f"{h:02d}:00" for h in range(0, 24)]
selected_hour_str = st.selectbox("Time (WIB)", hour_options, index=len(hour_options)-1)
selected_hour = int(selected_hour_str.split(":")[0])
start_datetime = datetime.combine(selected_date, time(selected_hour, 0, 0))
st.write(f"Start datetime (GMT+7): {start_datetime}")

# -----------------------------
# Instructions for upload
# -----------------------------
st.subheader("Instructions for Uploading Water Level Data")
st.info(
    f"- CSV must contain columns: 'Datetime' and 'Level Air'.\n"
    f"- 'Datetime' format: YYYY-MM-DD HH:MM:SS\n"
    f"- Data must cover **72 hours before the selected start datetime**: "
    f"{start_datetime - timedelta(hours=72)} to {start_datetime}\n"
    f"- Make sure there are no missing hours."
)

# -----------------------------
# Upload water level data
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV File (AWLR Joloi Logs)", type=["csv"])
wl_hourly = None
upload_success = False

if uploaded_file is not None:
    try:
        df_wl = pd.read_csv(uploaded_file, engine='python', skip_blank_lines=True)
        if "Datetime" not in df_wl.columns or "Level Air" not in df_wl.columns:
            st.error("The file must contain columns 'Datetime' and 'Level Air'.")
        else:
            # -----------------------
            # 1️⃣ Siapkan data
            # -----------------------
            df_wl["Datetime"] = pd.to_datetime(df_wl["Datetime"])
            df_wl = df_wl.sort_values("Datetime").set_index("Datetime")
            df_wl["Water_level"] = df_wl["Level Air"].clip(lower=0)  # hapus nilai negatif
            
            # -----------------------
            # 2️⃣ Deteksi spike singkat (<120 menit)
            # -----------------------
            df_wl['is_up'] = df_wl['Water_level'] > 0
            df_wl['group'] = (df_wl['is_up'] != df_wl['is_up'].shift()).cumsum()
            group_durations = df_wl.groupby('group').size() * 3  # durasi menit (3 menit per record)
            group_durations = group_durations.rename("duration_min")
            df_wl = df_wl.join(group_durations, on='group')
            
            short_spike = (df_wl['is_up']) & (df_wl['duration_min'] < 120)
            df_wl.loc[short_spike, 'Water_level'] = 0
            
            df_wl = df_wl.drop(columns=['is_up', 'group', 'duration_min', 'Level Air'])
            
            # -----------------------
            # 3️⃣ Interpolasi missing values
            # -----------------------
            df_wl['Water_level'] = df_wl['Water_level'].interpolate(method='time')
            
            # -----------------------
            # 4️⃣ Resample per jam
            # -----------------------
            wl_hourly = df_wl.resample('H').mean().reset_index()
            wl_hourly['Water_level'] = wl_hourly['Water_level'].interpolate().round(2)
            
            # Ambil hanya 72 jam terakhir sebelum start_datetime
            wl_hourly = wl_hourly[wl_hourly["Datetime"] >= (start_datetime - pd.Timedelta(hours=72))]
            
            # -----------------------
            # 5️⃣ Validasi missing hours (72 jam sebelum start)
            # -----------------------
            start_limit = start_datetime - pd.Timedelta(hours=72)
            end_limit = start_datetime
            expected_hours = pd.date_range(start=start_limit, end=end_limit - pd.Timedelta(hours=1), freq='H')
            actual_hours = pd.to_datetime(wl_hourly["Datetime"])
            missing_hours = sorted(set(expected_hours) - set(actual_hours))
            if missing_hours:
                missing_str = ', '.join([dt.strftime("%Y-%m-%d %H:%M") for dt in missing_hours])
                st.warning(f"The uploaded water level data is incomplete! Missing hours: {missing_str}")
            else:
                upload_success = True
                st.success("✅ File uploaded, cleaned, and validated successfully!")
                st.dataframe(wl_hourly)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

# -----------------------------
# Multi-point coordinates (actual Open-Meteo points)
# -----------------------------
directions = ["NW","N","NE","W","Center","E","SW","S","SE"]
points = [
    (0.38664, 113.64348),  # NW
    (0.38664, 114.13605),  # N
    (0.38664, 114.55825),  # NE
    (-0.10545, 113.56976), # W
    (-0.10545, 114.20109), # Center
    (-0.10545, 114.55183), # E
    (-0.59754, 113.62853), # SW
    (-0.59754, 114.12226), # S
    (-0.59754, 114.61599)  # SE
]
center = (-0.10545, 114.20109)  # tetap sebagai titik pusat IDW

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

numeric_cols = ["precipitation","cloud_cover","soil_moisture_0_to_7cm"]

def fetch_historical_multi(start_dt, end_dt):
    start_date, end_date = start_dt.date().isoformat(), end_dt.date().isoformat()
    all_dfs = []
    for lat, lon, dir_name in zip([p[0] for p in points],[p[1] for p in points], directions):
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=precipitation,cloud_cover,soil_moisture_0_to_7cm"
            f"&timezone=Asia%2FBangkok"
        )
        data = requests.get(url, timeout=30).json()
        df = pd.DataFrame(data["hourly"])
        df["latitude"], df["longitude"], df["direction"] = lat, lon, dir_name
        df["time"] = pd.to_datetime(df["time"])
        df["distance_km"] = haversine(df["latitude"], df["longitude"], center[0], center[1])
        all_dfs.append(df)
    # IDW
    weighted_list = []
    for time, group in pd.concat(all_dfs).groupby("time"):
        weights = 1 / (group["distance_km"]**2)
        weights /= weights.sum()
        weighted_vals = (group[numeric_cols].T*weights.values).T.sum()
        weighted_vals["Datetime"] = time
        weighted_list.append(weighted_vals)
    df_weighted = pd.DataFrame(weighted_list)
    df_weighted.rename(
        columns={
            "precipitation":"Rainfall",
            "cloud_cover":"Cloud_cover",
            "soil_moisture_0_to_7cm":"Soil_moisture"
        }, inplace=True
    )
    df_weighted = df_weighted[["Datetime","Rainfall","Cloud_cover","Soil_moisture"]]
    df_weighted[["Rainfall","Cloud_cover","Soil_moisture"]] = df_weighted[["Rainfall","Cloud_cover","Soil_moisture"]].round(2)
    return df_weighted

def fetch_forecast_multi():
    lats = ",".join([str(p[0]) for p in points])
    lons = ",".join([str(p[1]) for p in points])
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lats}&longitude={lons}"
        f"&hourly=precipitation,cloud_cover,soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,soil_moisture_3_to_9cm"
        f"&timezone=Asia%2FBangkok&forecast_days=7"
    )
    data = requests.get(url, timeout=30).json()
    all_dfs = []
    for i, dir_name in enumerate(directions):
        df_dict = {}
        for k, v in data["hourly"].items():
            # pastikan v[i] valid, jika hanya 1 titik, ambil v langsung
            if isinstance(v, list) or isinstance(v, np.ndarray):
                df_dict[k] = v[i] if len(v) > 1 else v[0]
            else:
                df_dict[k] = v
        df = pd.DataFrame(df_dict, index=[0])
        df["Datetime"] = pd.to_datetime(df["time"])
        df["direction"] = dir_name
        df["distance_km"] = haversine(points[i][0], points[i][1], center[0], center[1])
        all_dfs.append(df)
    # IDW
    weighted_list = []
    for time, group in pd.concat(all_dfs).groupby("Datetime"):
        sm_total = group[["soil_moisture_0_to_1cm","soil_moisture_1_to_3cm","soil_moisture_3_to_9cm"]].sum(axis=1)
        group["soil_moisture"] = sm_total
        weights = 1/(group["distance_km"]**2)
        weights /= weights.sum()
        weighted_vals = {
            "Rainfall": (group["precipitation"]*weights).sum(),
            "Cloud_cover": (group["cloud_cover"]*weights).sum(),
            "Soil_moisture": (group["soil_moisture"]*weights).sum(),
            "Datetime": time
        }
        weighted_list.append(weighted_vals)
    df_weighted = pd.DataFrame(weighted_list)
    df_weighted[["Rainfall","Cloud_cover","Soil_moisture"]] = df_weighted[["Rainfall","Cloud_cover","Soil_moisture"]].round(2)
    return df_weighted

# -----------------------------
# Run Forecast Button
# -----------------------------
run_forecast = st.button("Run 7-Day Forecast")
if "forecast_done" not in st.session_state:
    st.session_state["forecast_done"] = False
    st.session_state["final_df"] = None
    st.session_state["forecast_running"] = False

if run_forecast:
    st.session_state["forecast_done"] = False
    st.session_state["final_df"] = None
    st.session_state["forecast_running"] = True
    st.rerun()

# -----------------------------
# Forecast Logic
# -----------------------------
if upload_success and st.session_state["forecast_running"]:
    progress_container = st.empty()
    progress_bar = st.progress(0)
    step_counter, total_steps = 0, 4+168

    progress_container.markdown("Fetching historical climate data...")
    climate_hist = fetch_historical_multi(start_datetime - timedelta(hours=72), start_datetime)
    step_counter += 1
    progress_bar.progress(step_counter/total_steps)

    progress_container.markdown("Fetching forecast climate data...")
    climate_forecast = fetch_forecast_multi()
    step_counter += 1
    progress_bar.progress(step_counter/total_steps)

    progress_container.markdown("Merging water level and climate data...")
    merged_hist = pd.merge(wl_hourly, climate_hist, on="Datetime", how="left").sort_values("Datetime")
    merged_hist["Source"] = "Historical"

    forecast_hours = [start_datetime + timedelta(hours=i) for i in range(0,168)]
    forecast_df = pd.DataFrame({"Datetime":forecast_hours})
    forecast_merged = pd.merge(forecast_df, climate_forecast, on="Datetime", how="left")
    forecast_merged["Water_level"] = np.nan
    forecast_merged["Source"] = "Forecast"

    final_df = pd.concat([merged_hist, forecast_merged], ignore_index=True).sort_values("Datetime")
    final_df = final_df.apply(lambda x: np.round(x,2) if np.issubdtype(x.dtype, np.number) else x)
    step_counter += 1
    progress_bar.progress(step_counter/total_steps)

    progress_container.markdown("Forecasting water level 7 days iteratively...")
    model_features = model.get_booster().feature_names
    forecast_indices = final_df.index[final_df["Source"]=="Forecast"]

    for i, idx in enumerate(forecast_indices, start=1):
        step_counter += 1
        progress_bar.progress(step_counter/total_steps)
        X_forecast = pd.DataFrame(columns=model_features, index=[0])
        for f in model_features:
            base, lag = f.rsplit("_Lag",1)
            lag = int(lag)
            try:
                X_forecast.at[0,f] = final_df.loc[idx-lag, base]
            except:
                X_forecast.at[0,f] = final_df.loc[final_df["Source"]=="Historical", base].iloc[-lag]
        X_forecast = X_forecast.astype(float)
        y_hat = model.predict(X_forecast)[0]
        if y_hat < 0: y_hat = 0
        final_df.at[idx,"Water_level"] = round(y_hat,2)

    st.session_state["final_df"] = final_df
    st.session_state["forecast_done"] = True
    st.session_state["forecast_running"] = False
    progress_container.markdown("✅ 7-Day Water Level Forecast Completed!")
    progress_bar.progress(1.0)

# -----------------------------
# Display Forecast & Plot
# -----------------------------
result_container = st.empty()
if st.session_state["forecast_done"] and st.session_state["final_df"] is not None:
    final_df = st.session_state["final_df"]
    with result_container.container():
        st.subheader("Water Level + Climate Data with Forecast")
        def highlight_forecast(row):
            return ['background-color: #cfe9ff' if row['Source']=="Forecast" else '' for _ in row]
        styled_df = final_df.style.apply(highlight_forecast, axis=1).format("{:.2f}")
        st.dataframe(styled_df, use_container_width=True, height=500)

        # Plot
        st.subheader("Water Level Forecast Plot")
        fig = go.Figure()
        hist_df = final_df[final_df["Source"]=="Historical"]
        fore_df = final_df[final_df["Source"]=="Forecast"]

        fig.add_trace(go.Scatter(x=hist_df["Datetime"], y=hist_df["Water_level"],
                                 mode="lines+markers", name="Historical", line=dict(color="blue"), marker=dict(size=4)))
        if not fore_df.empty:
            last_val = hist_df["Water_level"].iloc[-1]
            forecast_x = pd.concat([pd.Series([hist_df["Datetime"].iloc[-1]]), fore_df["Datetime"]])
            forecast_y = pd.concat([pd.Series([last_val]), fore_df["Water_level"]])
            fig.add_trace(go.Scatter(x=forecast_x, y=forecast_y,
                                     mode="lines+markers", name="Forecast", line=dict(color="orange"), marker=dict(size=4)))

        fig.update_layout(title="Water Level Historical vs Forecast",
                          xaxis_title="Datetime", yaxis_title="Water Level (m)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Downloads
        # -----------------------------
        export_df = final_df[["Datetime","Water_level","Rainfall","Cloud_cover","Soil_moisture"]].copy()
        export_df["Datetime"] = export_df["Datetime"].astype(str)
        csv_buffer = export_df.to_csv(index=False).encode('utf-8')

        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name="Forecast")
        excel_buffer.seek(0)

        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=landscape(A4))
        styles = getSampleStyleSheet()
        data = [export_df.columns.tolist()] + export_df.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#007acc")),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),9),
            ('BOTTOMPADDING',(0,0),(-1,0),6),
            ('GRID',(0,0),(-1,-1),0.25,colors.grey),
        ]))
        elements = [Paragraph("Joloi Water Level Forecast", styles["Title"]), table]
        doc.build(elements)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("Download CSV", csv_buffer, "water_level_forecast.csv", "text/csv", use_container_width=True)
        with col2:
            st.download_button("Download Excel", excel_buffer, "water_level_forecast.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with col3:
            st.download_button("Download PDF", pdf_buffer.getvalue(), "water_level_forecast.pdf", "application/pdf", use_container_width=True)
else:
    result_container.empty()
