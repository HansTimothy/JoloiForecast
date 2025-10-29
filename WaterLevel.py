import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np
from datetime import datetime, timedelta, time
from xgboost import XGBRegressor
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
# Current time (GMT+7), rounded up to next full hour
# -----------------------------
now_utc = datetime.utcnow()
gmt7_now = now_utc + timedelta(hours=7)
rounded_now = (gmt7_now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0) \
    if gmt7_now.minute or gmt7_now.second or gmt7_now.microsecond else gmt7_now.replace(minute=0, second=0, microsecond=0)

# -----------------------------
# Select forecast start datetime
# -----------------------------
st.subheader("Select Start Date & Time for 7-Day Forecast")
selected_date = st.date_input("Date", value=rounded_now.date(), max_value=rounded_now.date())
hour_options = [f"{h:02d}:00" for h in range(0, rounded_now.hour + 1)] if selected_date == rounded_now.date() else [f"{h:02d}:00" for h in range(0, 24)]
selected_hour_str = st.selectbox("Time (WIB)", hour_options, index=len(hour_options)-1)
selected_hour = int(selected_hour_str.split(":")[0])
start_datetime = datetime.combine(selected_date, time(selected_hour,0,0))
st.write(f"Start datetime (GMT+7): {start_datetime}")

# -----------------------------
# Upload water level
# -----------------------------
st.subheader("Upload Hourly Water Level File")
uploaded_file = st.file_uploader("Upload CSV File (AWLR Joloi Logs)", type=["csv"])
wl_hourly = None
upload_success = False

if uploaded_file:
    df_wl = pd.read_csv(uploaded_file, engine='python', skip_blank_lines=True)
    if "Datetime" not in df_wl.columns or "Level Air" not in df_wl.columns:
        st.error("File must have columns 'Datetime' and 'Level Air'.")
    else:
        df_wl["Datetime"] = pd.to_datetime(df_wl["Datetime"]).dt.floor("H")
        start_limit = start_datetime - timedelta(hours=72)  # ✅ 72 jam
        df_wl_filtered = df_wl[(df_wl["Datetime"] >= start_limit) & (df_wl["Datetime"] < start_datetime)]
        expected_hours = pd.date_range(start=start_limit, end=start_datetime - timedelta(hours=1), freq='H')
        actual_hours = pd.to_datetime(df_wl_filtered["Datetime"].sort_values().unique())
        missing_hours = sorted(set(expected_hours) - set(actual_hours))
        if missing_hours:
            missing_str = ', '.join([dt.strftime("%Y-%m-%d %H:%M") for dt in missing_hours])
            st.warning(f"Data incomplete! Missing hours: {missing_str}")
        else:
            upload_success = True
            wl_hourly = df_wl_filtered.groupby("Datetime")["Level Air"].mean().reset_index().rename(columns={"Level Air": "Water_level"}).sort_values("Datetime").round(2)
            st.success("✅ File uploaded successfully!")
            st.dataframe(wl_hourly)

# -----------------------------
# Multi-point IDW climate fetch
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def fetch_climate_IDW(start_dt, end_dt, points, p=2):
    latitudes = ",".join([str(p[0]) for p in points])
    longitudes = ",".join([str(p[1]) for p in points])
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitudes}&longitude={longitudes}"
        f"&start_date={start_dt.date().isoformat()}&end_date={end_dt.date().isoformat()}"
        f"&hourly=precipitation,cloud_cover,soil_moisture_0_to_7cm&timezone=Asia%2FBangkok"
    )
    resp = requests.get(url, timeout=30)
    data = resp.json()
    all_dfs = []
    directions = ["NW","N","NE","W","Center","E","SW","S","SE"]
    if isinstance(data, list):
        for i, loc in enumerate(data):
            df = pd.DataFrame(loc["hourly"])
            df["latitude"] = loc["latitude"]
            df["longitude"] = loc["longitude"]
            df["direction"] = directions[i] if i < len(directions) else f"Point_{i+1}"
            all_dfs.append(df)
    elif isinstance(data, dict) and "hourly" in data:
        df = pd.DataFrame(data["hourly"])
        df["latitude"] = data.get("latitude", points[4][0])
        df["longitude"] = data.get("longitude", points[4][1])
        df["direction"] = "Center"
        all_dfs.append(df)
    else:
        raise ValueError("Format data Open-Meteo not recognized.")
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["time"] = pd.to_datetime(df_all["time"])
    df_all["distance_km"] = haversine(df_all["latitude"], df_all["longitude"], points[4][0], points[4][1])
    numeric_cols = ["precipitation","cloud_cover","soil_moisture_0_to_7cm"]
    weighted_list = []
    for t_, g in df_all.groupby("time"):
        w = 1/(g["distance_km"]**p)
        w /= w.sum()
        weighted = (g[numeric_cols].T * w.values).T.sum()
        weighted["Datetime"] = t_
        weighted_list.append(weighted)
    df_weighted = pd.DataFrame(weighted_list)
    df_weighted = df_weighted.rename(columns={"precipitation":"Rainfall","cloud_cover":"Cloud_cover","soil_moisture_0_to_7cm":"Soil_moisture"})
    df_weighted = df_weighted[["Datetime","Rainfall","Cloud_cover","Soil_moisture"]]
    df_weighted = df_weighted.round(2)
    return df_weighted

# -----------------------------
# Forecast Button
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
# Forecast Execution
# -----------------------------
if upload_success and st.session_state["forecast_running"]:
    progress_container = st.empty()
    progress_bar = st.progress(0)
    step_counter = 0
    total_steps = 3 + 168

    # 1️⃣ Climate historical 72 jam
    points = [
        (0.38664, 113.64348),(0.38664, 114.13605),(0.38664, 114.55825),
        (-0.10545, 113.56976),(-0.10545, 114.20109),(-0.10545, 114.55183),
        (-0.59754, 113.62853),(-0.59754, 114.12226),(-0.59754, 114.61599)
    ]
    start_hist = wl_hourly["Datetime"].min()
    end_hist = wl_hourly["Datetime"].max()
    climate_hist = fetch_climate_IDW(start_hist, end_hist, points)
    step_counter +=1
    progress_bar.progress(step_counter / total_steps)

    # 2️⃣ Merge water level + climate
    merged_df = pd.merge(wl_hourly, climate_hist, on="Datetime", how="left").sort_values("Datetime")
    merged_df["Source"] = "Historical"
    forecast_hours = [start_datetime + timedelta(hours=i) for i in range(168)]
    forecast_df = pd.DataFrame({"Datetime":forecast_hours})
    forecast_df["Water_level"] = np.nan
    forecast_df["Source"] = "Forecast"
    final_df = pd.concat([merged_df, forecast_df], ignore_index=True).sort_values("Datetime")
    final_df = final_df.round(2)
    step_counter +=1
    progress_bar.progress(step_counter / total_steps)

    # 3️⃣ Iterative forecast 7 hari
    model_features = model.get_booster().feature_names
    forecast_indices = final_df.index[final_df["Source"]=="Forecast"]
    for i, idx in enumerate(forecast_indices, start=1):
        step_counter +=1
        progress_bar.progress(step_counter / total_steps)
        progress_container.markdown(f"Forecasting hour {i}/{len(forecast_indices)}...")
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
        if y_hat < 0: y_hat = 0.0
        final_df.at[idx,"Water_level"] = round(y_hat,2)

    progress_container.markdown("✅ 7-Day Water Level Forecast Completed!")
    progress_bar.progress(1.0)
    st.session_state["final_df"] = final_df
    st.session_state["forecast_done"] = True
    st.session_state["forecast_running"] = False

# -----------------------------
# Display results
# -----------------------------
if st.session_state.get("forecast_done") and st.session_state["final_df"] is not None:
    final_df = st.session_state["final_df"]
    st.subheader("Water Level + Climate Data with Forecast")
    def highlight_forecast(row):
        color = 'background-color: #cfe9ff' if row['Source']=="Forecast" else ''
        return [color]*len(row)
    format_dict = {col: "{:.2f}" for col in final_df.select_dtypes(include=np.number).columns}
    styled_df = final_df.style.apply(highlight_forecast, axis=1).format(format_dict)
    st.dataframe(styled_df, use_container_width=True, height=500)

    # Plot
    st.subheader("Water Level Forecast Plot")
    fig = go.Figure()
    hist_df = final_df[final_df["Source"]=="Historical"]
    forecast_df_plot = final_df[final_df["Source"]=="Forecast"]

    fig.add_trace(go.Scatter(
        x=hist_df["Datetime"], y=hist_df["Water_level"],
        mode="lines+markers", name="Historical",
        line=dict(color="blue", width=2), marker=dict(size=4)
    ))
    if not forecast_df_plot.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df_plot["Datetime"], y=forecast_df_plot["Water_level"],
            mode="lines+markers", name="Forecast",
            line=dict(color="orange", width=2), marker=dict(size=4)
        ))
    fig.update_layout(xaxis_title="Datetime", yaxis_title="Water Level (m)",
                      title="Water Level Historical vs 7-Day Forecast",
                      template="plotly_white", hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)

    # Downloads
    export_df = final_df[["Datetime","Water_level"]].copy()
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
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#007acc")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
    ]))
    elements = [Paragraph("Joloi Water Level Forecast", styles["Title"]), table]
    doc.build(elements)

    col1, col2, col3 = st.columns(3)
    with col1: st.download_button("Download CSV", csv_buffer, "water_level_forecast.csv", "text/csv")
    with col2: st.download_button("Download Excel", excel_buffer, "water_level_forecast.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with col3: st.download_button("Download PDF", pdf_buffer.getvalue(), "water_level_forecast.pdf","application/pdf")
