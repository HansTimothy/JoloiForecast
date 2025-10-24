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

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("xgb_waterlevel_hourly_model.pkl")

st.title("🌊 Water Level Forecast Dashboard")

# -----------------------------
# Inisialisasi session_state
# -----------------------------
if "forecast_done" not in st.session_state:
    st.session_state["forecast_done"] = False
if "selected_datetime" not in st.session_state:
    st.session_state["selected_datetime"] = None
if "last_uploaded_name" not in st.session_state:
    st.session_state["last_uploaded_name"] = None

# -----------------------------
# Current time (GMT+7)
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

# -----------------------------
# Reset forecast jika user ganti tanggal/jam
# -----------------------------
if st.session_state["selected_datetime"] != start_datetime:
    st.session_state["selected_datetime"] = start_datetime
    st.session_state["forecast_done"] = False  # reset hasil forecast lama

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

# Reset forecast jika user upload file baru
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

            # Cek kelengkapan 24 jam terakhir
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
# Run 7-Day Forecast
# -----------------------------
run_forecast = st.button("Run 7-Day Forecast")

# Jika forecast sebelumnya sudah selesai, langsung tampilkan tanpa run ulang
if "forecast_done" not in st.session_state:
    st.session_state["forecast_done"] = False

# Jalankan forecast jika tombol ditekan
if upload_success and run_forecast:
    st.session_state["forecast_done"] = False  # reset flag dulu

    progress_container = st.empty()
    progress_bar = st.progress(0)
    
    total_steps = 3 + 168
    step_counter = 0

    # 1️⃣ Fetch climate data
    progress_container.markdown("Fetching climate data...")
    start_dt = wl_hourly["Datetime"].min()
    end_dt = wl_hourly["Datetime"].max()
    climate_hist = fetch_climate_historical(start_dt, end_dt)

    forecast_hours = [start_datetime + timedelta(hours=i) for i in range(0, 168)]
    forecast_df = pd.DataFrame({"Datetime": forecast_hours})
    forecast_start, forecast_end = forecast_df["Datetime"].min(), forecast_df["Datetime"].max()
    if forecast_end < gmt7_now:
        climate_forecast = fetch_climate_historical(forecast_start, forecast_end)
    elif forecast_start > gmt7_now:
        climate_forecast = fetch_climate_forecast()
    else:
        hist_df = fetch_climate_historical(forecast_start, gmt7_now)
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
    final_df = final_df.apply(lambda x: np.round(x,2) if np.issubdtype(x.dtype, np.number) else x)
    step_counter += 1
    progress_bar.progress(step_counter / total_steps)

    # 3️⃣ Iterative forecast
    progress_container.markdown("Forecasting water level for 7 days...")
    model_features = model.get_booster().feature_names
    forecast_indices = final_df.index[final_df["Source"]=="Forecast"]
    
    for i, idx in enumerate(forecast_indices, start=1):
        step_counter += 1
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
        final_df.at[idx, "Water_level"] = round(y_hat,2)

    progress_container.markdown("✅ 7-Day Water Level Forecast Completed!")
    progress_bar.progress(1.0)

    # Simpan hasil ke session_state
    st.session_state["final_df"] = final_df
    st.session_state["forecast_done"] = True


# -----------------------------
# Jika sudah pernah forecast, tampilkan hasilnya lagi tanpa run ulang
# -----------------------------
if st.session_state.get("forecast_done", False):

    final_df = st.session_state["final_df"]

    # Tabel hasil
    st.subheader("Water Level + Climate Data with Forecast")
    def highlight_forecast(row):
        color = 'background-color: #cfe9ff' if row['Source']=="Forecast" else ''
        return [color]*len(row)
    format_dict = {col: "{:.2f}" for col in final_df.select_dtypes(include=np.number).columns}
    styled_df = final_df.style.apply(highlight_forecast, axis=1).format(format_dict)
    st.dataframe(styled_df, use_container_width=True, height=500)

    # Plot
    import plotly.graph_objects as go
    st.subheader("Water Level Forecast Plot")
    rmse_est = 0.2
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=final_df.loc[final_df["Source"]=="Historical", "Datetime"],
        y=final_df.loc[final_df["Source"]=="Historical", "Water_level"],
        mode="lines+markers", name="Historical", line=dict(color="blue"), marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=final_df.loc[final_df["Source"]=="Forecast", "Datetime"],
        y=final_df.loc[final_df["Source"]=="Forecast", "Water_level"],
        mode="lines+markers", name="Forecast", line=dict(color="orange"), marker=dict(size=4)
    ))
    forecast_y = final_df.loc[final_df["Source"]=="Forecast", "Water_level"]
    forecast_x = final_df.loc[final_df["Source"]=="Forecast", "Datetime"]
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_x, forecast_x[::-1]]),
        y=pd.concat([forecast_y + rmse_est, (forecast_y - rmse_est).clip(0)[::-1]]),
        fill='toself', fillcolor='rgba(255,165,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip",
        showlegend=True, name="Forecast ± RMSE"
    ))
    fig.update_layout(
        xaxis_title="Datetime", yaxis_title="Water Level",
        title="Water Level Historical vs 7-Day Forecast",
        template="plotly_white", hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Tombol download (tidak reset data)
    # -----------------------------
    export_df = final_df[["Datetime", "Water_level"]].copy()
    export_df["Water_level"] = export_df["Water_level"].round(2)
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
    with col1:
        st.download_button("Download CSV", csv_buffer, "water_level_forecast.csv", "text/csv", use_container_width=True)
    with col2:
        st.download_button("Download Excel", excel_buffer, "water_level_forecast.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with col3:
        st.download_button("Download PDF", pdf_buffer.getvalue(), "water_level_forecast.pdf", "application/pdf", use_container_width=True)
