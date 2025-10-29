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
            
            # -----------------------
            # 5️⃣ Hanya 72 jam terakhir sebelum start_datetime, reset index
            # -----------------------
            wl_hourly = wl_hourly.loc[wl_hourly["Datetime"] >= (start_datetime - pd.Timedelta(hours=72))].reset_index(drop=True)
            
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

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

numeric_cols = ["precipitation","cloud_cover","soil_moisture_0_to_7cm"]

# -----------------------------
# Fungsi extract historical climate + IDW multi-point (1 API request)
# -----------------------------
# -----------------------------
# Fungsi extract historical climate + IDW multi-point
# -----------------------------
def fetch_historical_multi(start_dt, end_dt):
    """
    Mengambil data historis dari beberapa titik sekitar center,
    lalu melakukan IDW (Inverse Distance Weighting) per jam.
    """
    numeric_cols = ["precipitation","cloud_cover","soil_moisture_0_to_7cm"]
    latitudes = ",".join([str(p[0]) for p in points])
    longitudes = ",".join([str(p[1]) for p in points])

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitudes}&longitude={longitudes}"
        f"&start_date={start_dt.date().isoformat()}&end_date={end_dt.date().isoformat()}"
        f"&hourly=precipitation,cloud_cover,soil_moisture_0_to_7cm"
        f"&timezone=Asia%2FBangkok"
    )

    try:
        data = requests.get(url, timeout=60).json()
    except Exception as e:
        print("Error fetching data:", e)
        return pd.DataFrame()

    if "hourly" not in data or not data["hourly"]:
        return pd.DataFrame()

    times = pd.to_datetime(data["hourly"]["time"])
    all_dfs = []

    for i, (lat, lon, dir_name) in enumerate(zip([p[0] for p in points],
                                                 [p[1] for p in points],
                                                 directions)):
        try:
            df = pd.DataFrame({
                "Datetime": [v[i] for v in data["hourly"]["time"]],
                "precipitation": [v[i] for v in data["hourly"]["precipitation"]],
                "cloud_cover": [v[i] for v in data["hourly"]["cloud_cover"]],
                "soil_moisture_0_to_7cm": [v[i] for v in data["hourly"]["soil_moisture_0_to_7cm"]],
                "latitude": lat,
                "longitude": lon,
                "direction": dir_name
            })
        except Exception as e:
            print(f"Error processing point {dir_name}:", e)
            continue

        df["distance_km"] = haversine(lat, lon, center[0], center[1])
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    concat_df = pd.concat(all_dfs, ignore_index=True)

    weighted_list = []
    for time, group in concat_df.groupby("Datetime"):
        weights = 1 / (group["distance_km"]**2)
        weights /= weights.sum()
        weighted_vals = {
            "Datetime": time,
            "Rainfall": (group["precipitation"]*weights).sum(),
            "Cloud_cover": (group["cloud_cover"]*weights).sum(),
            "Soil_moisture": (group["soil_moisture_0_to_7cm"]*weights).sum()
        }
        weighted_list.append(weighted_vals)

    df_weighted = pd.DataFrame(weighted_list)
    df_weighted[["Rainfall","Cloud_cover","Soil_moisture"]] = df_weighted[["Rainfall","Cloud_cover","Soil_moisture"]].round(2)

    return df_weighted

# -----------------------------
# Fungsi extract forecast climate + IDW multi-point
# -----------------------------
def fetch_forecast_multi():
    """
    Mengambil data forecast 7 hari untuk beberapa titik sekitar center,
    lalu melakukan IDW (Inverse Distance Weighting) per jam.
    """
    numeric_cols = ["precipitation","cloud_cover","soil_moisture_0_1cm"]
    latitudes = ",".join([str(p[0]) for p in points])
    longitudes = ",".join([str(p[1]) for p in points])

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitudes}&longitude={longitudes}"
        f"&hourly=precipitation,cloud_cover,soil_moisture_0_1cm"
        f"&timezone=Asia%2FBangkok&forecast_days=7"
    )

    try:
        data = requests.get(url, timeout=60).json()
    except Exception as e:
        print("Error fetching forecast data:", e)
        return pd.DataFrame()

    if "hourly" not in data or not data["hourly"]:
        return pd.DataFrame()

    # Ganti kolom time → Datetime
    times = pd.to_datetime(data["hourly"]["time"])
    all_dfs = []

    for i, (lat, lon, dir_name) in enumerate(zip([p[0] for p in points],
                                                 [p[1] for p in points],
                                                 directions)):
        try:
            df = pd.DataFrame({
                "Datetime": [v[i] for v in data["hourly"]["time"]],
                "precipitation": [v[i] for v in data["hourly"]["precipitation"]],
                "cloud_cover": [v[i] for v in data["hourly"]["cloud_cover"]],
                "soil_moisture_0_1cm": [v[i] for v in data["hourly"]["soil_moisture_0_1cm"]],
                "latitude": lat,
                "longitude": lon,
                "direction": dir_name
            })
        except Exception as e:
            print(f"Error processing point {dir_name}:", e)
            continue

        df["distance_km"] = haversine(lat, lon, center[0], center[1])
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    concat_df = pd.concat(all_dfs, ignore_index=True)

    weighted_list = []
    for time, group in concat_df.groupby("Datetime"):
        weights = 1 / (group["distance_km"]**2)
        weights /= weights.sum()
        weighted_vals = {
            "Datetime": time,
            "Rainfall": (group["precipitation"]*weights).sum(),
            "Cloud_cover": (group["cloud_cover"]*weights).sum(),
            "Soil_moisture": (group["soil_moisture_0_1cm"]*weights).sum()
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
# Forecast Logic (Safe + Progress Detail)
# -----------------------------
if upload_success and st.session_state["forecast_running"]:
    progress_container = st.empty()
    
    total_forecast_hours = 168
    total_steps = 3 + total_forecast_hours  # 3 step sebelum loop + 168 jam forecast
    step_counter = 0
    progress_bar = st.progress(0)

    # 1️⃣ Fetch historical climate
    progress_container.markdown("Fetching historical climate data...")
    climate_hist = fetch_historical_multi(start_datetime - timedelta(hours=72), start_datetime)
    step_counter += 1
    progress_bar.progress(step_counter / total_steps)

    # 2️⃣ Fetch forecast climate
    progress_container.markdown("Fetching forecast climate data...")
    climate_forecast = fetch_forecast_multi()
    step_counter += 1
    progress_bar.progress(step_counter / total_steps)


    # 3️⃣ Merge water level and climate data
    progress_container.markdown("Merging water level and climate data...")
    
    # -----------------------------
    # Pastikan kolom waktu di semua DataFrame
    # -----------------------------
    for df_name, df in zip(["wl_hourly", "climate_hist", "climate_forecast"],
                           [wl_hourly, climate_hist, climate_forecast]):
        if df is not None:
            if "Datetime" not in df.columns and "time" in df.columns:
                df.rename(columns={"time": "Datetime"}, inplace=True)
    
    # Konversi ke datetime semua
    wl_hourly["Datetime"] = pd.to_datetime(wl_hourly["Datetime"])
    climate_hist["Datetime"] = pd.to_datetime(climate_hist["Datetime"])
    climate_forecast["Datetime"] = pd.to_datetime(climate_forecast["Datetime"])
    
    # Merge
    merged_hist = pd.merge(wl_hourly, climate_hist, on="Datetime", how="left").sort_values("Datetime")
    merged_hist["Source"] = "Historical"
    
    # Merge forecast
    forecast_hours = [start_datetime + timedelta(hours=i) for i in range(total_forecast_hours)]
    forecast_df = pd.DataFrame({"Datetime": forecast_hours})
    forecast_df["Datetime"] = pd.to_datetime(forecast_df["Datetime"])
    
    forecast_merged = pd.merge(forecast_df, climate_forecast, on="Datetime", how="left")
    forecast_merged["Water_level"] = np.nan
    forecast_merged["Source"] = "Forecast"
    
    # Gabungkan historical + forecast
    final_df = pd.concat([merged_hist, forecast_merged], ignore_index=True).sort_values("Datetime")
    final_df = final_df.apply(lambda x: np.round(x,2) if np.issubdtype(x.dtype, np.number) else x)
    
    step_counter += 1
    progress_bar.progress(step_counter / total_steps)

    # 4️⃣ Iterative forecast
    progress_container.markdown("Forecasting water level 7 days iteratively...")
    model_features = model.get_booster().feature_names
    forecast_indices = final_df.index[final_df["Source"]=="Forecast"]

    for i, idx in enumerate(forecast_indices, start=1):
        progress_container.markdown(f"Predicting hour {i}/{total_forecast_hours}...")
        X_forecast = pd.DataFrame(columns=model_features, index=[0])

        for f in model_features:
            if "_Lag" in f:
                base, lag_str = f.rsplit("_Lag",1)
                try:
                    lag = int(lag_str)
                except:
                    lag = 1
            else:
                base = f
                lag = 0

            # Ambil nilai lag dari final_df
            if base in final_df.columns:
                hist_values = final_df.loc[final_df["Source"]=="Historical", base]
                # Jika lag lebih besar dari panjang historical, ambil value pertama
                if idx-lag >= 0:
                    X_forecast.at[0,f] = final_df.iloc[idx-lag].get(base, 0)
                else:
                    X_forecast.at[0,f] = hist_values.iloc[0]
            else:
                # fallback jika kolom tidak ada
                X_forecast.at[0,f] = 0

        # pastikan tipe float
        X_forecast = X_forecast.astype(float)

        # prediksi
        y_hat = model.predict(X_forecast)[0]
        if y_hat < 0: y_hat = 0
        final_df.at[idx,"Water_level"] = round(y_hat,2)

        step_counter += 1
        progress_bar.progress(step_counter / total_steps)

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
        
        # Ambil semua kolom numerik
        numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Terapkan format hanya untuk kolom numerik
        styled_df = final_df.style.apply(highlight_forecast, axis=1)\
                                   .format({col: "{:.2f}" for col in numeric_cols})

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
