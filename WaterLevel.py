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
from scipy.signal import savgol_filter

# -----------------------------
# Helper: reset dashboard state (keadaan bersih)
# -----------------------------
def reset_dashboard_state(keep_uploaded=False):
    """Reset state so preview/results are cleared.
    If keep_uploaded True -> keep uploaded file in state (but clear results).
    """
    if not keep_uploaded:
        st.session_state.pop("uploaded_file_name", None)
        st.session_state.pop("wl_hourly", None)
        st.session_state.pop("upload_success", None)
    st.session_state["final_df"] = None
    st.session_state["forecast_done"] = False
    st.session_state["forecast_running"] = False

# -----------------------------
# Init session state keys (early)
# -----------------------------
if "forecast_done" not in st.session_state:
    st.session_state["forecast_done"] = False
if "forecast_running" not in st.session_state:
    st.session_state["forecast_running"] = False
if "final_df" not in st.session_state:
    st.session_state["final_df"] = None
if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None
if "wl_hourly" not in st.session_state:
    st.session_state["wl_hourly"] = None
if "upload_success" not in st.session_state:
    st.session_state["upload_success"] = False
if "last_date" not in st.session_state:
    st.session_state["last_date"] = None
if "last_hour" not in st.session_state:
    st.session_state["last_hour"] = None

# -----------------------------
# Title
# -----------------------------
st.title("🌊 Water Level Forecast Dashboard")

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
# Select start datetime (date + hour)
# -----------------------------
st.subheader("Select Start Date & Time for 7-Day Forecast")

selected_date = st.date_input(
    "Date",
    value=rounded_now.date(),
    max_value=rounded_now.date(),
    key="forecast_date_input"
)

if selected_date == rounded_now.date():
    hour_options = [f"{h:02d}:00" for h in range(0, rounded_now.hour + 1)]
else:
    hour_options = [f"{h:02d}:00" for h in range(0, 24)]

selected_hour_str = st.selectbox(
    "Time (WIB)",
    hour_options,
    index=len(hour_options) - 1,
    key="forecast_time_select"
)

selected_hour = int(selected_hour_str.split(":")[0])
start_datetime = datetime.combine(selected_date, time(selected_hour, 0, 0))
st.write(f"Start datetime (GMT+7): {start_datetime}")

# If date/hour changed -> clear preview and results
if (st.session_state.get("last_date") is None) or (st.session_state.get("last_hour") is None):
    st.session_state["last_date"] = selected_date
    st.session_state["last_hour"] = selected_hour
else:
    if (st.session_state["last_date"] != selected_date) or (st.session_state["last_hour"] != selected_hour):
        # user changed date/hour -> wipe uploaded preview + results
        reset_dashboard_state(keep_uploaded=False)
        st.session_state["last_date"] = selected_date
        st.session_state["last_hour"] = selected_hour
        # Immediately rerun so UI clears (prevents ghosting)
        st.experimental_rerun()

# -----------------------------
# Instructions
# -----------------------------
st.subheader("Instructions for Uploading Water Level Data")
st.info(
    f"Please upload a CSV file containing hourly water level data.\n"
    f"- The CSV must have columns: 'Datetime' and 'Level Air'.\n"
    f"- 'Datetime' should be in proper datetime format (e.g., YYYY-MM-DD HH:MM:SS).\n"
    f"- The data should cover the last 24 hours before the selected start datetime "
    f"({start_datetime - timedelta(hours=24)} to {start_datetime}).\n"
    f"- Make sure there are no missing hours in this period."
)

# -----------------------------
# Placeholders (so we can clear preview/result on demand)
# -----------------------------
preview_container = st.container()   # shows uploaded preview + Run button area
controls_col = None                  # used below to place Run button inline if needed
result_container = st.container()    # will hold forecast results (table + plot + downloads)

# -----------------------------
# Upload file
# -----------------------------
with preview_container:
    st.subheader("Upload Hourly Water Level File")
    uploaded_file = st.file_uploader("Upload CSV File (AWLR Joloi Logs)", type=["csv"], key="uploader")

    # If user replaced file (new name) -> reset states and re-read
    if uploaded_file is not None:
        if st.session_state.get("uploaded_file_name") != uploaded_file.name:
            # new upload: clear old results but keep this upload
            reset_dashboard_state(keep_uploaded=True)
            st.session_state["uploaded_file_name"] = uploaded_file.name

        # Try read and validate; store wl_hourly in session_state
        try:
            df_wl = pd.read_csv(uploaded_file, engine='python', skip_blank_lines=True)
            if "Datetime" not in df_wl.columns or "Level Air" not in df_wl.columns:
                st.error("The file must contain columns 'Datetime' and 'Level Air'.")
                st.session_state["upload_success"] = False
                st.session_state["wl_hourly"] = None
            else:
                df_wl["Datetime"] = pd.to_datetime(df_wl["Datetime"]).dt.floor("H")
                start_limit = start_datetime - pd.Timedelta(hours=24)
                df_wl_filtered = df_wl[(df_wl["Datetime"] >= start_limit) & (df_wl["Datetime"] < start_datetime)]

                expected_hours = pd.date_range(start=start_limit, end=start_datetime - pd.Timedelta(hours=1), freq='H')
                actual_hours = pd.to_datetime(df_wl_filtered["Datetime"].sort_values().unique())
                missing_hours = sorted(set(expected_hours) - set(actual_hours))
                if missing_hours:
                    missing_str = ', '.join([dt.strftime("%Y-%m-%d %H:%M") for dt in missing_hours])
                    st.warning(f"The uploaded water level data is incomplete! Missing hours: {missing_str}")
                    st.session_state["upload_success"] = False
                    st.session_state["wl_hourly"] = None
                else:
                    wl_hourly = (
                        df_wl_filtered.groupby("Datetime")["Level Air"].mean().reset_index()
                        .rename(columns={"Level Air": "Water_level"})
                        .sort_values(by="Datetime", ascending=True)
                        .round(2)
                    )
                    st.session_state["upload_success"] = True
                    st.session_state["wl_hourly"] = wl_hourly
                    # show preview here
                    st.success("✅ File uploaded successfully!")
                    st.dataframe(wl_hourly, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.session_state["upload_success"] = False
            st.session_state["wl_hourly"] = None
    else:
        # uploader empty: if previously had file name, it means user cleared uploader -> reset
        if st.session_state.get("uploaded_file_name"):
            reset_dashboard_state(keep_uploaded=False)
            st.session_state["uploaded_file_name"] = None
            # rerun so UI becomes clean
            st.experimental_rerun()
        else:
            st.info("No file uploaded yet.")

    # Always show Run button in preview area (enabled only if upload_success True)
    col_run, col_spacer = st.columns([1, 3])
    with col_run:
        run_forecast = st.button("Run 7-Day Forecast")
    # give user hint about whether button enabled
    if not st.session_state.get("upload_success", False):
        st.caption("Upload a valid 24-hour CSV to enable forecasting.")

# -----------------------------
# If user pressed Run -> clear preview/result and start forecast
# -----------------------------
# Note: do NOT call st.rerun before clearing UI containers to avoid ghost elements.
if 'run_forecast' in locals() and run_forecast:
    if st.session_state.get("upload_success", False) and st.session_state.get("wl_hourly") is not None:
        # Clear UI containers immediately to prevent showing stale plot/tables
        result_container.empty()
        preview_container.empty()
        # set flags and rerun to enter forecast-running branch
        st.session_state["forecast_done"] = False
        st.session_state["final_df"] = None
        st.session_state["forecast_running"] = True
        # experimental rerun is safer across versions to force fresh render
        st.experimental_rerun()
    else:
        st.warning("Please upload a valid 24-hour CSV file before running the forecast.")

# -----------------------------
# Forecast running block (executes after rerun when forecast_running True)
# -----------------------------
if st.session_state.get("forecast_running", False) and st.session_state.get("wl_hourly") is not None:
    # Show progress UI
    progress_container = st.empty()
    progress_bar = st.progress(0)
    progress_container.markdown("Fetching data and running forecast — please wait...")

    # total steps approx (fetch + merge + iterative forecast hours)
    total_steps = 3 + 168
    step_counter = 0

    # 1) fetch climate historical for wl period
    try:
        start_dt = st.session_state["wl_hourly"]["Datetime"].min()
        end_dt = st.session_state["wl_hourly"]["Datetime"].max()
    except Exception:
        # fallback if session stored DF differently
        start_dt = start_datetime - pd.Timedelta(hours=24)
        end_dt = start_datetime - pd.Timedelta(hours=1)

    climate_hist = fetch_climate_historical(start_dt, end_dt)
    step_counter += 1
    progress_bar.progress(step_counter / total_steps)

    # 2) prepare forecast climate data and merge
    forecast_hours = [start_datetime + timedelta(hours=i) for i in range(0, 168)]
    forecast_df = pd.DataFrame({"Datetime": forecast_hours})
    hist_df = fetch_climate_historical(forecast_df["Datetime"].min(), gmt7_now)
    fore_df = fetch_climate_forecast()
    climate_forecast = pd.concat([hist_df, fore_df]).drop_duplicates(subset="Datetime")
    step_counter += 1
    progress_bar.progress(step_counter / total_steps)

    # merge wl + climate
    merged_df = pd.merge(st.session_state["wl_hourly"], climate_hist, on="Datetime", how="left").sort_values("Datetime")
    merged_df["Source"] = "Historical"
    forecast_merged = pd.merge(forecast_df, climate_forecast, on="Datetime", how="left")
    forecast_merged["Water_level"] = np.nan
    forecast_merged["Source"] = "Forecast"
    final_df = pd.concat([merged_df, forecast_merged], ignore_index=True).sort_values("Datetime")
    final_df = final_df.apply(lambda x: np.round(x,2) if np.issubdtype(x.dtype, np.number) else x)
    step_counter += 1
    progress_bar.progress(step_counter / total_steps)

    # 3) iterative forecasting using model features (this part uses your loaded model)
    model_features = model.get_booster().feature_names
    forecast_indices = final_df.index[final_df["Source"] == "Forecast"]

    for i, idx in enumerate(forecast_indices, start=1):
        # build X using lag features from final_df; fallback to last historical if index-lag out-of-range
        X_forecast = pd.DataFrame(columns=model_features, index=[0])
        for f in model_features:
            base, lag = f.rsplit("_Lag", 1)
            lag = int(lag)
            try:
                X_forecast.at[0, f] = final_df.loc[idx - lag, base]
            except Exception:
                X_forecast.at[0, f] = final_df.loc[final_df["Source"] == "Historical", base].iloc[-lag]
        X_forecast = X_forecast.astype(float)
        y_hat = model.predict(X_forecast)[0]
        if y_hat < 0:
            y_hat = 0.0
        final_df.at[idx, "Water_level"] = round(y_hat, 2)

        # update progress occasionally to keep UI responsive
        if i % 8 == 0 or i == len(forecast_indices):
            step_counter += 8 if (step_counter + 8) <= total_steps else 1
            progress_bar.progress(min(step_counter / total_steps, 1.0))
            progress_container.markdown(f"Forecasting hour {i}/{len(forecast_indices)}...")

    progress_bar.progress(1.0)
    progress_container.markdown("✅ 7-Day Water Level Forecast Completed!")
    t.sleep(0.4)

    # smoothing
    final_df["Water_level_smooth"] = smooth_savgol(final_df["Water_level"], window=7, poly=2)
    historical_mask = final_df["Source"] == "Historical"
    final_df.loc[historical_mask, "Water_level_smooth"] = final_df.loc[historical_mask, "Water_level"]

    # store results & update flags
    st.session_state["final_df"] = final_df
    st.session_state["forecast_done"] = True
    st.session_state["forecast_running"] = False

    # clear progress UI then rerun to display results cleanly
    progress_container.empty()
    progress_bar.empty()
    st.experimental_rerun()

# -----------------------------
# Display results only when done (in result_container)
# -----------------------------
with result_container:
    if st.session_state.get("forecast_done", False) and st.session_state.get("final_df") is not None:
        final_df = st.session_state["final_df"]

        st.subheader("Water Level + Climate Data with Forecast (Smoothed)")

        def highlight_forecast(row):
            color = 'background-color: #cfe9ff' if row['Source'] == "Forecast" else ''
            return [color] * len(row)

        format_dict = {col: "{:.2f}" for col in final_df.select_dtypes(include=np.number).columns}
        styled_df = final_df.style.apply(highlight_forecast, axis=1).format(format_dict)
        st.dataframe(styled_df, use_container_width=True, height=500)

        # Plot
        st.subheader("Water Level Forecast Plot (Smoothed)")
        rmse_est = 0.06
        fig = go.Figure()
        hist_df = final_df[final_df["Source"] == "Historical"]
        forecast_df_plot = final_df[final_df["Source"] == "Forecast"]

        if not forecast_df_plot.empty:
            last_hist_time = hist_df["Datetime"].iloc[-1]
            last_hist_value = hist_df["Water_level_smooth"].iloc[-1]

            forecast_plot_x = pd.concat([pd.Series([last_hist_time]), forecast_df_plot["Datetime"]])
            forecast_plot_y = pd.concat([pd.Series([last_hist_value]), forecast_df_plot["Water_level_smooth"]])

            fig.add_trace(go.Scatter(
                x=forecast_plot_x,
                y=forecast_plot_y,
                mode="lines+markers",
                name="Forecast (Smoothed)",
                line=dict(color="orange", width=2),
                marker=dict(size=4),
                hovertemplate="Datetime: %{x}<br>Water Level: %{y:.2f} m"
            ))

            rmse_y_upper = (forecast_plot_y + rmse_est)
            rmse_y_lower = (forecast_plot_y - rmse_est).clip(0)
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_plot_x, forecast_plot_x[::-1]]),
                y=pd.concat([rmse_y_upper, rmse_y_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name=f"RMSE ±{rmse_est}"
            ))

        fig.add_trace(go.Scatter(
            x=hist_df["Datetime"],
            y=hist_df["Water_level_smooth"],
            mode="lines+markers",
            name="Historical",
            line=dict(color="blue", width=2),
            marker=dict(size=4),
            hovertemplate="Datetime: %{x}<br>Water Level: %{y:.2f} m"
        ))

        fig.update_layout(
            xaxis_title="Datetime",
            yaxis_title="Water Level (m)",
            title="Water Level Historical vs 7-Day Forecast (Smoothed)",
            template="plotly_white",
            hovermode="closest"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Downloads
        export_df = final_df[["Datetime", "Water_level", "Water_level_smooth"]].copy()
        export_df["Water_level"] = export_df["Water_level"].round(2)
        export_df["Water_level_smooth"] = export_df["Water_level_smooth"].round(2)
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
        elements = [Paragraph("Joloi Water Level Forecast (Smoothed)", styles["Title"]), table]
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
        # ensure nothing leftover shown
        st.write("")  # minimal placeholder, result area remains empty

