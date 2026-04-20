import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, warnings, joblib
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Beijing Air Quality",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 700; }
.block-container { padding-top: 1.5rem; }
.section-title {
    font-size: 1.4rem; font-weight: 700;
    color: #1f4e79;
    border-bottom: 2px solid #2196F3;
    padding-bottom: 4px; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────────────────────

STATION_FILES = {
    "Dongsi"        : "PRSA_Data_Dongsi_20130301-20170228.csv",
    "Wanshouxigong" : "PRSA_Data_Wanshouxigong_20130301-20170228.csv",
    "Dingling"      : "PRSA_Data_Dingling_20130301-20170228.csv",
    "Huairou"       : "PRSA_Data_Huairou_20130301-20170228.csv",
}

URBAN    = ["Dongsi", "Wanshouxigong"]
SUBURBAN = ["Dingling", "Huairou"]

POLL  = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
MET   = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
NUM   = POLL + MET

STATION_CLR = {
    "Dongsi"       : "#e74c3c",
    "Wanshouxigong": "#e67e22",
    "Dingling"     : "#27ae60",
    "Huairou"      : "#2980b9",
}
TYPE_CLR   = {"Urban": "#e74c3c", "Suburban": "#2980b9"}
SEASON_CLR = {"Spring":"#f39c12","Summer":"#27ae60",
              "Autumn":"#e67e22","Winter":"#3498db"}
SEASON_ORD = ["Spring","Summer","Autumn","Winter"]
MONTH_LBL  = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

AQI_CLR = {
    "Good"               : "#00c853",
    "Moderate"           : "#ffd600",
    "Lightly Polluted"   : "#ff6d00",
    "Moderately Polluted": "#dd2c00",
    "Heavily Polluted"   : "#6a1b9a",
    "Severely Polluted"  : "#3e0000",
}


# ────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────────

def get_aqi(v):
    if   v <= 35 : return "Good"
    elif v <= 75 : return "Moderate"
    elif v <= 115: return "Lightly Polluted"
    elif v <= 150: return "Moderately Polluted"
    elif v <= 250: return "Heavily Polluted"
    else         : return "Severely Polluted"


def get_season(m):
    if   m in [3,4,5]  : return "Spring"
    elif m in [6,7,8]  : return "Summer"
    elif m in [9,10,11]: return "Autumn"
    else               : return "Winter"


def time_of_day(h):
    if   6  <= h < 12: return "Morning"
    elif 12 <= h < 18: return "Afternoon"
    elif 18 <= h < 22: return "Evening"
    else             : return "Night"


# ────────────────────────────────────────────────────────────────
#  DATA LOADING — reads all 4 raw CSVs directly, no pre-processing needed
# ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading station data …")
def load_all():
    frames  = []
    missing = []

    for stn, fname in STATION_FILES.items():
        # check current directory and script directory
        candidates = [
            fname,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), fname),
        ]
        found = False
        for path in candidates:
            if os.path.exists(path):
                tmp = pd.read_csv(path, na_values="NA")
                # make sure station column is correct (some files have it already)
                tmp["station"] = stn
                frames.append(tmp)
                found = True
                break
        if not found:
            missing.append(stn)

    if not frames:
        return None, missing

    raw = pd.concat(frames, ignore_index=True)

    # build datetime column from year/month/day/hour
    raw["datetime"] = pd.to_datetime(raw[["year", "month", "day", "hour"]])

    # tag urban vs suburban
    raw["station_type"] = raw["station"].apply(
        lambda s: "Urban" if s in URBAN else "Suburban"
    )

    # sort chronologically per station before interpolating
    raw.sort_values(["station", "datetime"], inplace=True)
    raw.reset_index(drop=True, inplace=True)

    # interpolate missing sensor readings station by station
    raw[NUM] = (
        raw.groupby("station")[NUM]
        .transform(lambda x: x.interpolate(
            method="linear", limit_direction="both"
        ))
    )
    raw[NUM] = raw[NUM].ffill().bfill()

    # physical constraints — no negative concentrations or speeds
    for col in ["PM2.5","PM10","SO2","NO2","CO","O3","WSPM","RAIN"]:
        raw[col] = raw[col].clip(lower=0)

    # derived columns
    raw["season"]      = raw["month"].apply(get_season)
    raw["AQI_level"]   = raw["PM2.5"].apply(get_aqi)
    raw["day_of_week"] = raw["datetime"].dt.dayofweek
    raw["day_name"]    = raw["datetime"].dt.day_name()
    raw["is_weekend"]  = raw["day_of_week"].isin([5, 6]).astype(int)
    raw["time_of_day"] = raw["hour"].apply(time_of_day)

    return raw, missing


# ────────────────────────────────────────────────────────────────
#  SIDEBAR
# ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌏 Navigation")
    page = st.radio("", [
        "🏠 Home",
        "📂 Dataset Explorer",
        "📊 Visualisations",
        "🤖 Model & Predictions"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "**Module:** CMP7005  \n**Assessment:** PRAC1  \n"
        "**Dataset:** Beijing Air Quality  \n"
        "**Stations:** Dongsi, Wanshouxigong (Urban) "
        "| Dingling, Huairou (Suburban)"
    )
    st.markdown("---")


# ────────────────────────────────────────────────────────────────
#  LOAD DATA
# ────────────────────────────────────────────────────────────────

df_full, missing_stns = load_all()

if df_full is None:
    st.error(
        "❌ No CSV files found. Place all four "
        "PRSA_Data_*.csv files in the same folder as app.py and restart."
    )
    st.stop()

if missing_stns:
    st.warning(
        f"⚠️ Files not found for: {', '.join(missing_stns)}. "
        "Dashboard running with available stations only."
    )

# sidebar filters — only rendered after successful load
with st.sidebar:
    st.markdown("### 🔧 Filters")
    avail_stns = sorted(df_full["station"].unique().tolist())
    sel_stns   = st.multiselect("Stations", avail_stns, default=avail_stns)

    avail_yrs = sorted(df_full["year"].unique().tolist())
    yr_min, yr_max = int(avail_yrs[0]), int(avail_yrs[-1])
    yr_range = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))

# apply filters
df = df_full[
    df_full["station"].isin(sel_stns) &
    df_full["year"].between(yr_range[0], yr_range[1])
].copy()


# ════════════════════════════════════════════════════════════════
#  PAGE — HOME
# ════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.markdown(
        "<h2 style='text-align:center;color:#1f4e79'>🌏 Beijing Air Quality Dashboard</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;color:#888'>CMP7005 PRAC1 — "
        "Interactive Data Exploration & PM2.5 Prediction Platform</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # top KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📍 Stations",   df["station"].nunique())
    k2.metric("📋 Records",    f"{len(df):,}")
    k3.metric("📅 Years",      f"{df['year'].min()}–{df['year'].max()}")
    k4.metric("💨 Avg PM2.5",  f"{df['PM2.5'].mean():.1f} μg/m³")

    st.markdown("---")
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<p class="section-title">PM2.5 Monthly Trend by Station</p>',
                    unsafe_allow_html=True)
        mon = (
            df.groupby(["year","month","station"])["PM2.5"]
            .mean().reset_index()
        )
        mon["date"] = pd.to_datetime(mon[["year","month"]].assign(day=1))
        fig = px.line(
            mon, x="date", y="PM2.5", color="station",
            color_discrete_map=STATION_CLR,
            labels={"PM2.5":"PM2.5 (μg/m³)", "date":""},
            template="plotly_white"
        )
        fig.add_hline(y=35, line_dash="dot", line_color="green",
                      annotation_text="Good (35 μg/m³)")
        fig.add_hline(y=75, line_dash="dot", line_color="orange",
                      annotation_text="Moderate (75 μg/m³)")
        fig.update_layout(legend_title="Station", margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<p class="section-title">AQI Level Distribution</p>',
                    unsafe_allow_html=True)
        aqi_cnt = df["AQI_level"].value_counts()
        aqi_ord = [a for a in
                   ["Good","Moderate","Lightly Polluted","Moderately Polluted",
                    "Heavily Polluted","Severely Polluted"]
                   if a in aqi_cnt.index]
        aqi_cnt = aqi_cnt.reindex(aqi_ord)
        fig2 = px.pie(
            values=aqi_cnt.values, names=aqi_cnt.index,
            color=aqi_cnt.index, color_discrete_map=AQI_CLR,
            template="plotly_white"
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label",
                           textfont_size=10)
        fig2.update_layout(showlegend=False, margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # summary box
    u_avg = df[df["station_type"]=="Urban"]["PM2.5"].mean()
    s_avg = df[df["station_type"]=="Suburban"]["PM2.5"].mean()
    st.info(
        f"📌 Urban stations average **{u_avg:.1f} μg/m³** PM2.5 "
        f"vs suburban **{s_avg:.1f} μg/m³** — a gap of **{u_avg-s_avg:.1f} μg/m³**. "
        "Use the sidebar to filter by station or year."
    )


# ════════════════════════════════════════════════════════════════
#  PAGE — DATASET EXPLORER
# ════════════════════════════════════════════════════════════════

elif page == "📂 Dataset Explorer":
    st.markdown('<p class="section-title">📂 Dataset Explorer</p>',
                unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["📄 Raw Data", "📊 Statistics", "🔍 Missing Values"])

    with t1:
        keep_cols = ["datetime","station","station_type","PM2.5","PM10",
                     "SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN",
                     "WSPM","season","AQI_level"]
        keep_cols = [c for c in keep_cols if c in df.columns]

        n   = st.slider("Rows to display", 10, 1000, 100)
        srt = st.selectbox("Sort by", keep_cols)
        st.dataframe(df[keep_cols].sort_values(srt).head(n),
                     use_container_width=True)

        st.download_button(
            "⬇️ Download filtered CSV",
            data=df[keep_cols].to_csv(index=False).encode(),
            file_name="filtered_air_quality.csv",
            mime="text/csv"
        )

    with t2:
        pick = st.selectbox("Station", ["All"] + avail_stns)
        sub  = df if pick == "All" else df[df["station"]==pick]
        desc = sub[NUM].describe().T.round(2)
        desc["skewness"] = sub[NUM].skew().round(3)
        desc["kurtosis"] = sub[NUM].kurt().round(3)
        st.dataframe(desc, use_container_width=True)

        st.markdown("##### Mean values per station")
        st.dataframe(df.groupby("station")[NUM].mean().round(2),
                     use_container_width=True)

    with t3:
        # post-processing nulls (should be near zero)
        mis     = df[NUM].isnull().sum()
        mis_pct = (mis / len(df) * 100).round(2)
        st.dataframe(
            pd.DataFrame({"Count": mis, "%": mis_pct})
            .sort_values("%", ascending=False),
            use_container_width=True
        )
        if mis.sum() == 0:
            st.success("✅ No missing values after interpolation.")


# ════════════════════════════════════════════════════════════════
#  PAGE — VISUALISATIONS
# ════════════════════════════════════════════════════════════════

elif page == "📊 Visualisations":
    st.markdown('<p class="section-title">📊 Data Visualisations</p>',
                unsafe_allow_html=True)

    t_dist, t_rel, t_time, t_comp = st.tabs([
        "📦 Distributions", "🔗 Relationships",
        "📅 Temporal",      "🏙️ Urban vs Suburban"
    ])

    # ── distributions ────────────────────────────────────────────
    with t_dist:
        var  = st.selectbox("Variable", NUM, key="dist_var")
        unit = {"PM2.5":"μg/m³","PM10":"μg/m³","SO2":"μg/m³","NO2":"μg/m³",
                "CO":"μg/m³","O3":"μg/m³","TEMP":"°C","PRES":"hPa",
                "DEWP":"°C","RAIN":"mm","WSPM":"m/s"}.get(var, "")

        ca, cb = st.columns(2)
        with ca:
            fig = px.histogram(
                df, x=var, color="station", nbins=80,
                barmode="overlay", opacity=0.65,
                color_discrete_map=STATION_CLR,
                title=f"{var} histogram ({unit})",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        with cb:
            fig = px.box(
                df, x="station", y=var, color="station_type",
                color_discrete_map=TYPE_CLR,
                title=f"{var} by station",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = px.violin(
            df, x="season", y=var, color="station_type",
            box=True, points=False,
            color_discrete_map=TYPE_CLR,
            category_orders={"season": SEASON_ORD},
            title=f"{var} by season",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── relationships ─────────────────────────────────────────────
    with t_rel:
        ca, cb = st.columns(2)
        xv = ca.selectbox("X axis", NUM,
                          index=NUM.index("TEMP") if "TEMP" in NUM else 0)
        yv = cb.selectbox("Y axis", NUM,
                          index=NUM.index("PM2.5") if "PM2.5" in NUM else 1)

        samp = df.sample(min(6000, len(df)), random_state=7)
        fig  = px.scatter(
            samp, x=xv, y=yv, color="station",
            color_discrete_map=STATION_CLR,
            opacity=0.45, trendline="ols",
            title=f"{yv} vs {xv}",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Pearson Correlation Matrix")
        corr = df[NUM].corr().round(2)
        fig2 = px.imshow(
            corr, text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto",
            template="plotly_white"
        )
        fig2.update_layout(height=520)
        st.plotly_chart(fig2, use_container_width=True)

    # ── temporal ──────────────────────────────────────────────────
    with t_time:
        tv  = st.selectbox("Variable", NUM, key="time_var")
        agg = st.radio("Group by",
                       ["Monthly","Seasonal","Hourly","Day of Week"],
                       horizontal=True)

        if agg == "Monthly":
            tmp = (
                df.groupby(["year","month","station"])[tv]
                .mean().reset_index()
            )
            tmp["date"] = pd.to_datetime(tmp[["year","month"]].assign(day=1))
            fig = px.line(tmp, x="date", y=tv, color="station",
                          color_discrete_map=STATION_CLR,
                          title=f"Monthly {tv}",
                          template="plotly_white")

        elif agg == "Seasonal":
            tmp = df.groupby(["season","station"])[tv].mean().reset_index()
            fig = px.bar(tmp, x="season", y=tv, color="station",
                         barmode="group",
                         category_orders={"season": SEASON_ORD},
                         color_discrete_map=STATION_CLR,
                         title=f"Seasonal {tv}",
                         template="plotly_white")

        elif agg == "Hourly":
            tmp = df.groupby(["hour","station"])[tv].mean().reset_index()
            fig = px.line(tmp, x="hour", y=tv, color="station",
                          markers=True,
                          color_discrete_map=STATION_CLR,
                          title=f"Diurnal profile — {tv}",
                          template="plotly_white")
            fig.update_xaxes(tickvals=list(range(0,24,2)))

        else:
            day_ord = ["Monday","Tuesday","Wednesday",
                       "Thursday","Friday","Saturday","Sunday"]
            tmp = df.groupby(["day_name","station"])[tv].mean().reset_index()
            fig = px.bar(tmp, x="day_name", y=tv, color="station",
                         barmode="group",
                         category_orders={"day_name": day_ord},
                         color_discrete_map=STATION_CLR,
                         title=f"{tv} by day of week",
                         template="plotly_white")

        fig.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # ── urban vs suburban ─────────────────────────────────────────
    with t_comp:
        cv = st.selectbox("Variable", NUM, key="comp_var")

        # seasonal comparison bars
        ts = df.groupby(["season","station_type"])[cv].mean().reset_index()
        fig = px.bar(
            ts, x="season", y=cv, color="station_type",
            barmode="group",
            category_orders={"season": SEASON_ORD},
            color_discrete_map=TYPE_CLR,
            title=f"Urban vs Suburban — {cv} by season",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # radar — normalised pollutant profile
        st.markdown("##### Normalised Pollutant Profile")
        prof   = df.groupby("station")[POLL].mean()
        prof_n = (prof - prof.min()) / (prof.max() - prof.min())

        fig3 = go.Figure()
        for stn in prof_n.index:
            v = prof_n.loc[stn].tolist() + [prof_n.loc[stn, POLL[0]]]
            cats = POLL + [POLL[0]]
            fig3.add_trace(go.Scatterpolar(
                r=v, theta=cats, fill="toself", name=stn,
                line_color=STATION_CLR.get(stn, "#aaa")
            ))
        fig3.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
            template="plotly_white",
            title="Normalised pollutant profile by station"
        )
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  PAGE — MODEL & PREDICTIONS
# ════════════════════════════════════════════════════════════════

elif page == "🤖 Model & Predictions":
    st.markdown('<p class="section-title">🤖 PM2.5 Prediction — Random Forest</p>',
                unsafe_allow_html=True)

    t_perf, t_pred, t_fi = st.tabs(
        ["📈 Performance", "🔮 Predict", "🎯 Feature Importance"]
    )

    with t_perf:
        if not os.path.exists("rf_pm25_model.pkl"):
            st.warning(
                "⚠️ Model file not found. "
                "Run Task 3 in the Jupyter notebook to train and save it."
            )
        else:
            if os.path.exists("model_metrics.csv"):
                met  = pd.read_csv("model_metrics.csv")
                cols = st.columns(len(met))
                for i, row in met.iterrows():
                    cols[i].metric(row["Metric"], row["Value"])

            if os.path.exists("predictions.csv"):
                pred_df = pd.read_csv("predictions.csv")
                samp    = pred_df.sample(min(2500, len(pred_df)), random_state=3)

                fig = px.scatter(
                    samp, x="actual", y="predicted", opacity=0.45,
                    title="Actual vs Predicted PM2.5",
                    labels={"actual":"Actual (μg/m³)",
                            "predicted":"Predicted (μg/m³)"},
                    template="plotly_white"
                )
                mx = max(samp["actual"].max(), samp["predicted"].max())
                fig.add_shape(type="line", x0=0,y0=0,x1=mx,y1=mx,
                              line=dict(color="red", dash="dash"))
                st.plotly_chart(fig, use_container_width=True)

                pred_df["residual"] = pred_df["actual"] - pred_df["predicted"]
                fig2 = px.histogram(
                    pred_df, x="residual", nbins=80,
                    title="Residual distribution",
                    template="plotly_white"
                )
                st.plotly_chart(fig2, use_container_width=True)

    with t_pred:
        if not (os.path.exists("rf_pm25_model.pkl") and
                os.path.exists("scaler.pkl")):
            st.warning("⚠️ Model or scaler file missing. Run the notebook first.")
        else:
            model  = joblib.load("rf_pm25_model.pkl")
            scaler = joblib.load("scaler.pkl")

            st.markdown("Adjust the inputs below and press **Predict**.")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Pollutants**")
                pm10 = st.slider("PM10 (μg/m³)",  0.0, 800.0,  80.0)
                so2  = st.slider("SO2 (μg/m³)",   0.0, 500.0,  20.0)
                no2  = st.slider("NO2 (μg/m³)",   0.0, 400.0,  60.0)
                co   = st.slider("CO (μg/m³)",    0.0, 10000.0,1200.0)
                o3   = st.slider("O3 (μg/m³)",    0.0, 400.0,  50.0)

            with c2:
                st.markdown("**Meteorology**")
                temp = st.slider("Temp (°C)",        -20.0, 42.0,   15.0)
                pres = st.slider("Pressure (hPa)",   985.0, 1045.0, 1013.0)
                dewp = st.slider("Dew Point (°C)",   -40.0, 30.0,   5.0)
                rain = st.slider("Rain (mm)",         0.0,  70.0,   0.0)
                wspm = st.slider("Wind speed (m/s)",  0.0,  13.0,   2.0)

            with c3:
                st.markdown("**Time & Location**")
                hr   = st.slider("Hour (0–23)",         0, 23, 12)
                mo   = st.slider("Month (1–12)",         1, 12,  6)
                dow  = st.slider("Day of week (0=Mon)",  0,  6,  2)
                wknd = st.checkbox("Is weekend?")
                stype= st.selectbox("Station type", ["Urban","Suburban"])

            # season encoding matching notebook
            seas_enc = {12:0,1:0,2:0, 3:1,4:1,5:1,
                        6:2,7:2,8:2, 9:3,10:3,11:3}.get(mo, 1)
            type_enc = 1 if stype == "Urban" else 0

            X_in = np.array([[pm10, so2, no2, co, o3,
                               temp, pres, dewp, rain, wspm,
                               hr, mo, dow, int(wknd),
                               type_enc, seas_enc]])

            if st.button("🔮 Predict PM2.5", type="primary"):
                try:
                    raw_pred = model.predict(scaler.transform(X_in))[0]
                    pred_val = float(max(0, raw_pred))

                    if   pred_val <= 35 : lvl,col = "Good",               "#00c853"
                    elif pred_val <= 75 : lvl,col = "Moderate",           "#ffd600"
                    elif pred_val <= 115: lvl,col = "Lightly Polluted",   "#ff6d00"
                    elif pred_val <= 150: lvl,col = "Moderately Polluted","#dd2c00"
                    elif pred_val <= 250: lvl,col = "Heavily Polluted",   "#6a1b9a"
                    else                : lvl,col = "Severely Polluted",  "#3e0000"

                    st.markdown(
                        f"<div style='background:{col}22;border:2px solid {col};"
                        f"border-radius:8px;padding:1.2rem;text-align:center;"
                        f"margin:1rem 0'>"
                        f"<h2 style='color:{col};margin:0'>"
                        f"Predicted PM2.5: {pred_val:.1f} μg/m³</h2>"
                        f"<h3 style='margin:0.4rem 0'>AQI Level: {lvl}</h3>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                except Exception as err:
                    st.error(f"Prediction error: {err}")

    with t_fi:
        if os.path.exists("feature_importance.csv"):
            fi  = pd.read_csv("feature_importance.csv").sort_values("importance")
            fig = px.bar(
                fi, x="importance", y="feature",
                orientation="h", color="importance",
                color_continuous_scale="Blues",
                title="Feature Importance — Random Forest",
                template="plotly_white"
            )
            fig.update_layout(height=480, showlegend=False,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Higher score = greater influence on predictions. "
                "PM10 and CO typically dominate as they are co-emitted "
                "from combustion sources alongside PM2.5."
            )
        else:
            st.info("Run Task 3 in the notebook to generate feature_importance.csv.")
