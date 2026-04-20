"""
CMP7005 - Programming for Data Analysis
PRAC1: From Data to Application Development
Task 4: Interactive Streamlit Application

Author: [Your Student ID]
Description: Interactive dashboard for Beijing Air Quality data exploration,
             visualisation, and PM2.5 prediction using Random Forest model.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Beijing Air Quality Dashboard",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a3a5c;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a3a5c;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  DATA & MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load the preprocessed dataset."""
    csv_path = "beijing_air_quality_processed.csv"
    if not os.path.exists(csv_path):
        st.error("❌ Dataset not found. Please run the notebook first to generate 'beijing_air_quality_processed.csv'.")
        return None
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    return df

@st.cache_resource
def load_model():
    """Load the trained Random Forest model."""
    model_path = "rf_pm25_model.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

df = load_data()
model = load_model()

# AQI colour palette
AQI_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Lightly Polluted": "#ff7e00",
    "Moderately Polluted": "#ff0000",
    "Heavily Polluted": "#8f3f97",
    "Severely Polluted": "#7e0023",
}

POLLUTANT_UNITS = {
    "PM2.5": "μg/m³", "PM10": "μg/m³",
    "SO2": "μg/m³",  "NO2": "μg/m³",
    "CO": "μg/m³",   "O3": "μg/m³",
    "TEMP": "°C",    "PRES": "hPa",
    "DEWP": "°C",    "RAIN": "mm",
    "WSPM": "m/s",
}


# ─────────────────────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌏 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home", "📂 Dataset Explorer", "📊 Visualisations", "🤖 Model & Predictions"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "**Module:** CMP7005  \n"
        "**Assessment:** PRAC1  \n"
        "**Dataset:** Beijing Air Quality  \n"
        "**Stations:** Dongsi, Wanshouxigong (Urban) | Dingling, Huairou (Suburban)"
    )
    st.markdown("---")

    if df is not None:
        st.markdown("### 🔧 Global Filters")
        selected_stations = st.multiselect(
            "Stations",
            options=df["station"].unique().tolist(),
            default=df["station"].unique().tolist()
        )
        year_range = st.slider(
            "Year Range",
            int(df["year"].min()), int(df["year"].max()),
            (int(df["year"].min()), int(df["year"].max()))
        )
        # Apply filters
        df_filtered = df[
            (df["station"].isin(selected_stations)) &
            (df["year"].between(*year_range))
        ]
    else:
        df_filtered = None


# ─────────────────────────────────────────────────────────────
#  PAGE: HOME
# ─────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<p class="main-title">🌏 Beijing Air Quality Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">CMP7005 PRAC1 — Interactive Data Exploration & PM2.5 Prediction Platform</p>',
        unsafe_allow_html=True
    )

    if df is not None:
        # KPI metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📍 Stations", df_filtered["station"].nunique())
        with col2:
            st.metric("📋 Total Records", f"{len(df_filtered):,}")
        with col3:
            st.metric("📅 Date Range", f"{df['year'].min()}–{df['year'].max()}")
        with col4:
            avg_pm25 = df_filtered["PM2.5"].mean()
            st.metric("💨 Avg PM2.5", f"{avg_pm25:.1f} μg/m³")

        st.markdown("---")

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown('<p class="section-header">📈 PM2.5 Trend Overview</p>', unsafe_allow_html=True)
            monthly_avg = (
                df_filtered
                .groupby(["year", "month", "station"])["PM2.5"]
                .mean()
                .reset_index()
            )
            monthly_avg["date"] = pd.to_datetime(
                monthly_avg[["year", "month"]].assign(day=1)
            )
            fig = px.line(
                monthly_avg, x="date", y="PM2.5", color="station",
                title="Monthly Average PM2.5 by Station",
                labels={"PM2.5": "PM2.5 (μg/m³)", "date": "Date"},
                template="plotly_white"
            )
            fig.add_hline(y=35, line_dash="dash", line_color="green",
                          annotation_text="Good (35 μg/m³)")
            fig.add_hline(y=75, line_dash="dash", line_color="orange",
                          annotation_text="Moderate (75 μg/m³)")
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown('<p class="section-header">🎯 AQI Distribution</p>', unsafe_allow_html=True)
            aqi_counts = df_filtered["AQI_level"].value_counts()
            aqi_order = ["Good", "Moderate", "Lightly Polluted",
                         "Moderately Polluted", "Heavily Polluted", "Severely Polluted"]
            aqi_counts = aqi_counts.reindex([x for x in aqi_order if x in aqi_counts.index])
            fig2 = px.pie(
                values=aqi_counts.values,
                names=aqi_counts.index,
                color=aqi_counts.index,
                color_discrete_map=AQI_COLORS,
                title="AQI Level Distribution",
                template="plotly_white"
            )
            fig2.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        <div class="info-box">
        <b>🔑 Key Findings at a Glance</b><br>
        This dataset covers hourly air quality measurements from 4 Beijing stations (2013–2017).
        Urban stations (Dongsi, Wanshouxigong) consistently record higher PM2.5 than suburban
        stations (Dingling, Huairou). Winter months show the highest pollution due to heating
        emissions and lower wind speeds. Navigate using the sidebar to explore the data, charts,
        and model predictions in detail.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please run the Jupyter notebook first to generate the data files.")


# ─────────────────────────────────────────────────────────────
#  PAGE: DATASET EXPLORER
# ─────────────────────────────────────────────────────────────
elif page == "📂 Dataset Explorer":
    st.markdown('<p class="section-header">📂 Dataset Explorer</p>', unsafe_allow_html=True)

    if df_filtered is not None:
        tab1, tab2, tab3 = st.tabs(["📄 Raw Data", "📊 Summary Statistics", "🔍 Missing Values"])

        with tab1:
            st.markdown(f"Showing **{len(df_filtered):,}** rows from selected filters.")
            col1, col2 = st.columns(2)
            with col1:
                rows_to_show = st.slider("Rows to display", 10, 500, 100)
            with col2:
                sort_col = st.selectbox("Sort by", df_filtered.columns.tolist(), index=0)

            display_cols = [
                "datetime", "station", "station_type", "PM2.5", "PM10",
                "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP",
                "RAIN", "WSPM", "season", "AQI_level"
            ]
            display_cols = [c for c in display_cols if c in df_filtered.columns]
            st.dataframe(
                df_filtered[display_cols].sort_values(sort_col).head(rows_to_show),
                use_container_width=True
            )

            # Download
            csv_data = df_filtered[display_cols].to_csv(index=False)
            st.download_button(
                "⬇️ Download Filtered Data as CSV",
                data=csv_data,
                file_name="filtered_air_quality.csv",
                mime="text/csv"
            )

        with tab2:
            st.markdown("### Descriptive Statistics by Station")
            numeric_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                            "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
            numeric_cols = [c for c in numeric_cols if c in df_filtered.columns]
            stat_station = st.selectbox("Select station", ["All"] + df_filtered["station"].unique().tolist())

            if stat_station == "All":
                stat_df = df_filtered[numeric_cols]
            else:
                stat_df = df_filtered[df_filtered["station"] == stat_station][numeric_cols]

            st.dataframe(stat_df.describe().round(2), use_container_width=True)

            st.markdown("### Station Comparison — Mean Values")
            station_means = df_filtered.groupby("station")[numeric_cols].mean().round(2)
            st.dataframe(station_means, use_container_width=True)

        with tab3:
            st.markdown("### Missing Values Analysis")
            missing = df_filtered.isnull().sum()
            missing_pct = (missing / len(df_filtered) * 100).round(2)
            missing_table = pd.DataFrame({
                "Missing Count": missing,
                "Missing (%)": missing_pct
            }).sort_values("Missing (%)").ascending=False

            missing_table = pd.DataFrame({
                "Missing Count": missing,
                "Missing (%)": missing_pct
            }).sort_values("Missing (%)", ascending=False)

            st.dataframe(missing_table, use_container_width=True)

            if missing.sum() == 0:
                st.success("✅ No missing values in the filtered dataset (preprocessing already applied).")
    else:
        st.warning("No data loaded.")


# ─────────────────────────────────────────────────────────────
#  PAGE: VISUALISATIONS
# ─────────────────────────────────────────────────────────────
elif page == "📊 Visualisations":
    st.markdown('<p class="section-header">📊 Data Visualisations</p>', unsafe_allow_html=True)

    if df_filtered is not None:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📦 Distribution", "🔗 Relationships", "📅 Temporal Trends", "🏙️ Station Comparison"
        ])

        # ── TAB 1: DISTRIBUTIONS ──
        with tab1:
            st.markdown("### Univariate Distribution Analysis")
            numeric_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                            "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
            numeric_cols = [c for c in numeric_cols if c in df_filtered.columns]
            var = st.selectbox("Select variable", numeric_cols)
            unit = POLLUTANT_UNITS.get(var, "")

            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    df_filtered, x=var, color="station",
                    nbins=80, barmode="overlay", opacity=0.7,
                    title=f"Distribution of {var} ({unit})",
                    labels={var: f"{var} ({unit})"},
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.box(
                    df_filtered, x="station", y=var, color="station_type",
                    title=f"Box Plot: {var} by Station",
                    labels={var: f"{var} ({unit})"},
                    template="plotly_white"
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Seasonal distribution
            fig3 = px.violin(
                df_filtered, x="season", y=var, color="station_type",
                box=True, points=False,
                category_orders={"season": ["Spring", "Summer", "Autumn", "Winter"]},
                title=f"Seasonal Distribution of {var}",
                template="plotly_white"
            )
            st.plotly_chart(fig3, use_container_width=True)

        # ── TAB 2: RELATIONSHIPS ──
        with tab2:
            st.markdown("### Bivariate & Multivariate Analysis")
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("X axis", numeric_cols, index=numeric_cols.index("TEMP") if "TEMP" in numeric_cols else 0)
            with col2:
                y_var = st.selectbox("Y axis", numeric_cols, index=numeric_cols.index("PM2.5") if "PM2.5" in numeric_cols else 1)

            sample_size = min(5000, len(df_filtered))
            scatter_df = df_filtered.sample(sample_size, random_state=42)

            fig = px.scatter(
                scatter_df, x=x_var, y=y_var,
                color="station", opacity=0.5,
                trendline="ols",
                title=f"{y_var} vs {x_var}",
                labels={
                    x_var: f"{x_var} ({POLLUTANT_UNITS.get(x_var, '')})",
                    y_var: f"{y_var} ({POLLUTANT_UNITS.get(y_var, '')})"
                },
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Correlation Heatmap")
            corr_cols = [c for c in numeric_cols if c in df_filtered.columns]
            corr_matrix = df_filtered[corr_cols].corr().round(2)
            fig_heat = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Pearson Correlation Matrix",
                template="plotly_white",
                aspect="auto"
            )
            fig_heat.update_layout(height=550)
            st.plotly_chart(fig_heat, use_container_width=True)

        # ── TAB 3: TEMPORAL TRENDS ──
        with tab3:
            st.markdown("### Temporal Analysis")
            trend_var = st.selectbox("Variable for temporal analysis", numeric_cols)
            agg_level = st.radio("Aggregation", ["Monthly", "Seasonal", "Hourly", "Day of Week"], horizontal=True)

            if agg_level == "Monthly":
                temp_df = (
                    df_filtered.groupby(["year", "month", "station"])[trend_var]
                    .mean().reset_index()
                )
                temp_df["date"] = pd.to_datetime(temp_df[["year", "month"]].assign(day=1))
                fig = px.line(
                    temp_df, x="date", y=trend_var, color="station",
                    title=f"Monthly Average {trend_var}",
                    labels={trend_var: f"{trend_var} ({POLLUTANT_UNITS.get(trend_var, '')})"},
                    template="plotly_white"
                )

            elif agg_level == "Seasonal":
                temp_df = df_filtered.groupby(["season", "station"])[trend_var].mean().reset_index()
                fig = px.bar(
                    temp_df, x="season", y=trend_var, color="station",
                    barmode="group",
                    category_orders={"season": ["Spring", "Summer", "Autumn", "Winter"]},
                    title=f"Seasonal Average {trend_var}",
                    labels={trend_var: f"{trend_var} ({POLLUTANT_UNITS.get(trend_var, '')})"},
                    template="plotly_white"
                )

            elif agg_level == "Hourly":
                temp_df = df_filtered.groupby(["hour", "station"])[trend_var].mean().reset_index()
                fig = px.line(
                    temp_df, x="hour", y=trend_var, color="station",
                    markers=True,
                    title=f"Average {trend_var} by Hour of Day",
                    labels={
                        "hour": "Hour (0-23)",
                        trend_var: f"{trend_var} ({POLLUTANT_UNITS.get(trend_var, '')})"
                    },
                    template="plotly_white"
                )

            else:  # Day of Week
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                temp_df = df_filtered.groupby(["day_name", "station"])[trend_var].mean().reset_index()
                fig = px.bar(
                    temp_df, x="day_name", y=trend_var, color="station",
                    barmode="group",
                    category_orders={"day_name": day_order},
                    title=f"Average {trend_var} by Day of Week",
                    labels={trend_var: f"{trend_var} ({POLLUTANT_UNITS.get(trend_var, '')})"},
                    template="plotly_white"
                )

            st.plotly_chart(fig, use_container_width=True)

        # ── TAB 4: STATION COMPARISON ──
        with tab4:
            st.markdown("### Urban vs Suburban Comparison")
            comp_var = st.selectbox("Variable", numeric_cols, key="comp_var")

            # Mean by type
            type_means = df_filtered.groupby(["station_type", "season"])[comp_var].mean().reset_index()
            fig = px.bar(
                type_means, x="season", y=comp_var, color="station_type",
                barmode="group",
                category_orders={"season": ["Spring", "Summer", "Autumn", "Winter"]},
                title=f"Urban vs Suburban {comp_var} by Season",
                labels={comp_var: f"{comp_var} ({POLLUTANT_UNITS.get(comp_var, '')})"},
                template="plotly_white",
                color_discrete_map={"Urban": "#e74c3c", "Suburban": "#2ecc71"}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Radar chart — pollutant profile per station
            st.markdown("### Pollutant Profile Radar Chart")
            radar_cols = ["PM2.5", "PM10", "SO2", "NO2", "O3"]
            radar_cols = [c for c in radar_cols if c in df_filtered.columns]
            station_profile = df_filtered.groupby("station")[radar_cols].mean()
            # Normalise 0–1
            station_profile_norm = (station_profile - station_profile.min()) / (
                station_profile.max() - station_profile.min()
            )

            fig_radar = go.Figure()
            for stn in station_profile_norm.index:
                vals = station_profile_norm.loc[stn].tolist()
                vals += [vals[0]]  # close loop
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=radar_cols + [radar_cols[0]],
                    fill="toself", name=stn
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Normalised Pollutant Profile by Station",
                template="plotly_white"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.warning("No data loaded.")


# ─────────────────────────────────────────────────────────────
#  PAGE: MODEL & PREDICTIONS
# ─────────────────────────────────────────────────────────────
elif page == "🤖 Model & Predictions":
    st.markdown('<p class="section-header">🤖 PM2.5 Prediction Model</p>', unsafe_allow_html=True)

    if df_filtered is not None:
        tab1, tab2, tab3 = st.tabs(["📈 Model Performance", "🔮 Manual Prediction", "🎯 Feature Importance"])

        # ── TAB 1: MODEL PERFORMANCE ──
        with tab1:
            st.markdown("### Random Forest Regressor — PM2.5 Prediction")

            if model is None:
                st.warning("⚠️ Model file 'rf_pm25_model.pkl' not found. Please run the notebook to train and save the model first.")
            else:
                # Load test metrics if available
                metrics_path = "model_metrics.csv"
                if os.path.exists(metrics_path):
                    metrics_df = pd.read_csv(metrics_path)
                    st.markdown("#### Performance Metrics")
                    cols = st.columns(len(metrics_df))
                    for i, row in metrics_df.iterrows():
                        if i < len(cols):
                            cols[i].metric(row["Metric"], row["Value"])
                else:
                    st.info("Run the notebook to generate model metrics.")

                # Actual vs Predicted plot
                pred_path = "predictions.csv"
                if os.path.exists(pred_path):
                    pred_df = pd.read_csv(pred_path)
                    st.markdown("#### Actual vs Predicted PM2.5 (Test Set)")
                    sample_pred = pred_df.sample(min(2000, len(pred_df)), random_state=42)
                    fig = px.scatter(
                        sample_pred, x="actual", y="predicted",
                        opacity=0.5,
                        title="Actual vs Predicted PM2.5",
                        labels={"actual": "Actual PM2.5 (μg/m³)", "predicted": "Predicted PM2.5 (μg/m³)"},
                        template="plotly_white"
                    )
                    # Perfect prediction line
                    max_val = max(sample_pred["actual"].max(), sample_pred["predicted"].max())
                    fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                                  line=dict(color="red", dash="dash"))
                    st.plotly_chart(fig, use_container_width=True)

                    # Residuals
                    pred_df["residual"] = pred_df["actual"] - pred_df["predicted"]
                    fig2 = px.histogram(
                        pred_df, x="residual", nbins=80,
                        title="Residual Distribution",
                        labels={"residual": "Residual (Actual − Predicted)"},
                        template="plotly_white"
                    )
                    st.plotly_chart(fig2, use_container_width=True)

        # ── TAB 2: MANUAL PREDICTION ──
        with tab2:
            st.markdown("### 🔮 Predict PM2.5 from Input Conditions")
            st.markdown("Adjust the meteorological and pollutant parameters to predict PM2.5.")

            if model is None:
                st.warning("⚠️ Model not loaded. Run the notebook first.")
            else:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Pollutants**")
                    pm10_val  = st.slider("PM10 (μg/m³)",   0.0, 800.0, 80.0)
                    so2_val   = st.slider("SO2 (μg/m³)",    0.0, 500.0, 20.0)
                    no2_val   = st.slider("NO2 (μg/m³)",    0.0, 400.0, 60.0)
                    co_val    = st.slider("CO (μg/m³)",     0.0, 10000.0, 1200.0)
                    o3_val    = st.slider("O3 (μg/m³)",     0.0, 400.0, 50.0)

                with col2:
                    st.markdown("**Meteorology**")
                    temp_val  = st.slider("Temperature (°C)", -20.0, 42.0, 15.0)
                    pres_val  = st.slider("Pressure (hPa)",   985.0, 1045.0, 1013.0)
                    dewp_val  = st.slider("Dew Point (°C)",   -40.0, 30.0, 5.0)
                    rain_val  = st.slider("Rain (mm)",        0.0, 70.0, 0.0)
                    wspm_val  = st.slider("Wind Speed (m/s)", 0.0, 13.0, 2.0)

                with col3:
                    st.markdown("**Time & Location**")
                    hour_val  = st.slider("Hour (0-23)",  0, 23, 12)
                    month_val = st.slider("Month (1-12)", 1, 12, 6)
                    dow_val   = st.slider("Day of Week (0=Mon)", 0, 6, 2)
                    weekend_val = st.checkbox("Is Weekend?")
                    stn_type  = st.selectbox("Station Type", ["Urban", "Suburban"])

                season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                              6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
                stn_type_enc = 1 if stn_type == "Urban" else 0
                season_enc = season_map.get(month_val, 1)

                input_features = np.array([[
                    pm10_val, so2_val, no2_val, co_val, o3_val,
                    temp_val, pres_val, dewp_val, rain_val, wspm_val,
                    hour_val, month_val, dow_val, int(weekend_val),
                    stn_type_enc, season_enc
                ]])

                if st.button("🔮 Predict PM2.5", type="primary"):
                    try:
                        prediction = model.predict(input_features)[0]
                        prediction = max(0, prediction)

                        # AQI classification
                        if prediction <= 35:    aqi = "Good";              color = "#00e400"
                        elif prediction <= 75:  aqi = "Moderate";          color = "#ffff00"
                        elif prediction <= 115: aqi = "Lightly Polluted";  color = "#ff7e00"
                        elif prediction <= 150: aqi = "Moderately Polluted"; color = "#ff0000"
                        elif prediction <= 250: aqi = "Heavily Polluted";  color = "#8f3f97"
                        else:                   aqi = "Severely Polluted"; color = "#7e0023"

                        st.markdown(f"""
                        <div style="background-color:{color}22; border:2px solid {color};
                                    border-radius:10px; padding:1.5rem; text-align:center; margin:1rem 0;">
                            <h2 style="color:{color}; margin:0;">Predicted PM2.5: {prediction:.1f} μg/m³</h2>
                            <h3 style="margin:0.5rem 0;">AQI Level: {aqi}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Prediction error: {e}")

        # ── TAB 3: FEATURE IMPORTANCE ──
        with tab3:
            st.markdown("### 🎯 Feature Importance")
            fi_path = "feature_importance.csv"
            if os.path.exists(fi_path):
                fi_df = pd.read_csv(fi_path).sort_values("importance", ascending=True)
                fig = px.bar(
                    fi_df, x="importance", y="feature",
                    orientation="h",
                    title="Random Forest Feature Importance (PM2.5 Prediction)",
                    labels={"importance": "Importance Score", "feature": "Feature"},
                    color="importance",
                    color_continuous_scale="Blues",
                    template="plotly_white"
                )
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="info-box">
                <b>📌 Interpretation:</b> Features with higher importance scores have greater influence 
                on PM2.5 predictions. PM10 and CO typically rank highest due to their co-emission 
                with PM2.5 from combustion sources. Meteorological variables like temperature and 
                pressure also play significant roles in pollutant dispersion.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Feature importance file not found. Please run the notebook first.")
    else:
        st.warning("No data loaded.")
