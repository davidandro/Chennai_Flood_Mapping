import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
from datetime import datetime, timedelta
import shap
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="FloodWatch Chennai",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stAppViewContainer"] { background: #0a0f1e; }
[data-testid="stSidebar"] { background: #0d1428; border-right: 1px solid #1e2d4a; }
.hero-title { font-family:'Space Mono',monospace; font-size:2.4rem; font-weight:700; color:#e8f4fd; letter-spacing:-1px; line-height:1.1; margin-bottom:0.2rem; }
.hero-sub { font-size:0.95rem; color:#5b8fb9; margin-bottom:1rem; }
.live-badge { display:inline-block; background:#052e16; color:#22c55e; border:1px solid #22c55e55; border-radius:999px; font-family:'Space Mono',monospace; font-size:0.7rem; padding:0.2rem 0.8rem; letter-spacing:1px; margin-left:0.8rem; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
.metric-card { background:#111827; border:1px solid #1e3a5f; border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:0.8rem; }
.metric-label { font-size:0.7rem; color:#4a7fa5; text-transform:uppercase; letter-spacing:1.5px; font-weight:600; margin-bottom:0.3rem; }
.metric-value { font-family:'Space Mono',monospace; font-size:1.8rem; font-weight:700; line-height:1; }
.zone-low      { color:#22c55e; } .zone-medium { color:#f59e0b; }
.zone-high     { color:#f97316; } .zone-veryhigh { color:#ef4444; }
.badge-low      { background:#052e16; color:#22c55e; border:1px solid #22c55e44; border-radius:999px; padding:0.3rem 1rem; font-family:'Space Mono',monospace; font-size:0.78rem; font-weight:700; }
.badge-medium   { background:#1c1408; color:#f59e0b; border:1px solid #f59e0b44; border-radius:999px; padding:0.3rem 1rem; font-family:'Space Mono',monospace; font-size:0.78rem; font-weight:700; }
.badge-high     { background:#1c0d02; color:#f97316; border:1px solid #f9731644; border-radius:999px; padding:0.3rem 1rem; font-family:'Space Mono',monospace; font-size:0.78rem; font-weight:700; }
.badge-veryhigh { background:#1f0505; color:#ef4444; border:1px solid #ef444444; border-radius:999px; padding:0.3rem 1rem; font-family:'Space Mono',monospace; font-size:0.78rem; font-weight:700; }
.section-header { font-family:'Space Mono',monospace; font-size:0.72rem; color:#4a7fa5; text-transform:uppercase; letter-spacing:2px; font-weight:700; border-bottom:1px solid #1e3a5f; padding-bottom:0.5rem; margin-bottom:1rem; }
.info-box  { background:#0d1f35; border-left:3px solid #2563eb; border-radius:0 8px 8px 0; padding:0.8rem 1rem; margin:0.5rem 0; font-size:0.87rem; color:#93c5fd; }
.warn-box  { background:#1c1408; border-left:3px solid #f59e0b; border-radius:0 8px 8px 0; padding:0.8rem 1rem; margin:0.5rem 0; font-size:0.87rem; color:#fcd34d; }
.danger-box{ background:#1f0505; border-left:3px solid #ef4444; border-radius:0 8px 8px 0; padding:0.8rem 1rem; margin:0.5rem 0; font-size:0.87rem; color:#fca5a5; }
.safe-box  { background:#052e16; border-left:3px solid #22c55e; border-radius:0 8px 8px 0; padding:0.8rem 1rem; margin:0.5rem 0; font-size:0.87rem; color:#86efac; }
.weather-stat { display:flex; justify-content:space-between; padding:0.4rem 0; border-bottom:1px solid #1e2d4a; font-size:0.83rem; color:#94a3b8; }
.weather-stat span:last-child { font-family:'Space Mono',monospace; color:#38bdf8; }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    'Total_Rainfall','Max_Daily_Rainfall','Rainy_Days','Avg_Rainfall',
    'Avg_Temp','Avg_Humidity','Avg_Wind','Avg_Baro',
    'City_Inundation_Risk','Stagnation_Risk',
    'Is_NE_Monsoon','Is_SW_Monsoon','Rain_Lag1','Rain_Lag2'
]

CHENNAI_ZONES = {
    "North Chennai (Tondiarpet, Royapuram)":  (13.1167, 80.2833),
    "Central Chennai (T.Nagar, Nungambakkam)":(13.0604, 80.2496),
    "South Chennai (Adyar, Velachery)":       (13.0067, 80.2206),
    "West Chennai (Ambattur, Poonamallee)":   (13.0983, 80.1683),
    "Suburban (Tambaram, Chromepet)":         (12.9249, 80.1000),
    "Coastal (Besant Nagar, Marina)":         (13.0002, 80.2721),
}

# ── Live weather from Open-Meteo (FREE, no API key) ──
@st.cache_data(ttl=1800)   # refresh every 30 minutes
def fetch_live_weather(lat, lon):
    try:
        # Current weather
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m",
            "hourly": "precipitation",
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "Asia/Kolkata",
            "forecast_days": 7,
            "past_days": 30,
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        current = data.get("current", {})
        daily   = data.get("daily",   {})
        hourly  = data.get("hourly",  {})

        # Last 30 days rainfall → monthly aggregates
        hourly_precip = hourly.get("precipitation", [])
        total_rain_30d = sum(hourly_precip) if hourly_precip else 150
        max_daily = max([sum(hourly_precip[i:i+24]) for i in range(0, min(len(hourly_precip),720), 24)] or [40])
        rainy_days = sum(1 for i in range(0, min(len(hourly_precip),720), 24)
                        if sum(hourly_precip[i:i+24]) > 1)

        # 7-day forecast
        forecast_dates  = daily.get("time", [])
        forecast_rain   = daily.get("precipitation_sum", [0]*7)
        forecast_tmax   = daily.get("temperature_2m_max", [30]*7)
        forecast_prob   = daily.get("precipitation_probability_max", [30]*7)

        return {
            "temp":        current.get("temperature_2m", 28),
            "humidity":    current.get("relative_humidity_2m", 75),
            "wind":        current.get("wind_speed_10m", 15),
            "pressure":    current.get("surface_pressure", 1008),
            "precip_now":  current.get("precipitation", 0),
            "total_rain":  round(total_rain_30d, 1),
            "max_daily":   round(max_daily, 1),
            "rainy_days":  rainy_days,
            "avg_rain":    round(total_rain_30d / max(rainy_days, 1), 1),
            "forecast_dates": forecast_dates,
            "forecast_rain":  forecast_rain,
            "forecast_tmax":  forecast_tmax,
            "forecast_prob":  forecast_prob,
            "fetched_at":  datetime.now().strftime("%d %b %Y, %H:%M IST"),
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e),
                "temp":28,"humidity":75,"wind":15,"pressure":1008,
                "precip_now":0,"total_rain":150,"max_daily":40,
                "rainy_days":8,"avg_rain":18,"fetched_at":"unavailable",
                "forecast_dates":[],"forecast_rain":[],"forecast_tmax":[],"forecast_prob":[]}

@st.cache_resource
def load_flood_model():
    try:
        m = load_model("cnn_lstm_chennai_flood.h5", compile=False)
        m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return m
    except:
        return None

@st.cache_resource
def get_scaler():
    np.random.seed(42)
    dummy = pd.DataFrame({
        'Total_Rainfall':np.random.exponential(120,400).clip(0,900),
        'Max_Daily_Rainfall':np.random.exponential(60,400).clip(0,450),
        'Rainy_Days':np.random.randint(1,31,400).astype(float),
        'Avg_Rainfall':np.random.exponential(20,400).clip(0,100),
        'Avg_Temp':np.random.normal(28,3,400).clip(18,40),
        'Avg_Humidity':np.random.normal(75,12,400).clip(30,100),
        'Avg_Wind':np.random.normal(15,6,400).clip(0,60),
        'Avg_Baro':np.random.normal(1010,5,400).clip(990,1030),
        'City_Inundation_Risk':np.ones(400)*1.43,
        'Stagnation_Risk':np.ones(400)*1.2,
        'Is_NE_Monsoon':np.random.randint(0,2,400).astype(float),
        'Is_SW_Monsoon':np.random.randint(0,2,400).astype(float),
        'Rain_Lag1':np.random.exponential(100,400).clip(0,700),
        'Rain_Lag2':np.random.exponential(90,400).clip(0,600),
    })
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(); sc.fit(dummy[FEATURES].values); return sc

def mc_predict(model, X, n=50):
    preds = np.array([model(X, training=True).numpy().flatten() for _ in range(n)])
    return float(preds.mean()), float(preds.std())

def risk_zone(prob):
    if prob<0.25:  return "Low",      "badge-low",      "zone-low",      "🟢"
    elif prob<0.50: return "Medium",  "badge-medium",   "zone-medium",   "🟡"
    elif prob<0.75: return "High",    "badge-high",     "zone-high",     "🟠"
    else:           return "Very High","badge-veryhigh","zone-veryhigh", "🔴"

def conf_label(s):
    if s<0.08:  return "Very High","#22c55e"
    elif s<0.15: return "High",    "#86efac"
    elif s<0.22: return "Medium",  "#f59e0b"
    else:        return "Low",     "#ef4444"

# ══════════════ SIDEBAR ══════════════
with st.sidebar:
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:1.3rem;font-weight:700;color:#e8f4fd;">🌊 FloodWatch</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.82rem;color:#5b8fb9;margin-bottom:1rem;">Chennai Flood Susceptibility</div>', unsafe_allow_html=True)
    st.markdown("---")

    zone_name = st.selectbox("📍 Select Chennai Zone", list(CHENNAI_ZONES.keys()))
    lat, lon = CHENNAI_ZONES[zone_name]

    st.markdown("---")
    st.markdown("**Data source**")
    data_mode = st.radio("", ["🌐 Live (Open-Meteo API)", "✏️ Manual input"], label_visibility="collapsed")
    st.markdown("---")

    # Always fetch live
    weather = fetch_live_weather(lat, lon)

    if data_mode == "✏️ Manual input":
        st.markdown('<div style="font-size:0.72rem;color:#4a7fa5;text-transform:uppercase;letter-spacing:2px;font-weight:700;margin-bottom:0.5rem;">🌧️ Rainfall</div>', unsafe_allow_html=True)
        total_rain  = st.slider("Total Monthly Rainfall (mm)", 0, 900, int(weather["total_rain"]))
        max_rain    = st.slider("Max Daily Rainfall (mm)",      0, 400, int(weather["max_daily"]))
        rainy_days  = st.slider("Rainy Days",                   0,  31, int(weather["rainy_days"]))
        lag1        = st.slider("Prev Month Rainfall (mm)",     0, 700, 180)
        lag2        = st.slider("2-Month Prior Rainfall (mm)",  0, 600,  90)
        st.markdown('<div style="font-size:0.72rem;color:#4a7fa5;text-transform:uppercase;letter-spacing:2px;font-weight:700;margin:0.5rem 0;">🌡️ Weather</div>', unsafe_allow_html=True)
        avg_temp    = st.slider("Temperature (°C)",   18, 40, int(weather["temp"]))
        avg_hum     = st.slider("Humidity (%)",        30,100, int(weather["humidity"]))
        avg_wind    = st.slider("Wind Speed (km/h)",    0, 60, int(weather["wind"]))
        avg_baro    = st.slider("Pressure (hPa)",     990,1030,int(weather["pressure"]))
    else:
        total_rain = weather["total_rain"]; max_rain = weather["max_daily"]
        rainy_days = weather["rainy_days"]; lag1 = weather["total_rain"] * 0.8
        lag2 = weather["total_rain"] * 0.5; avg_temp = weather["temp"]
        avg_hum = weather["humidity"]; avg_wind = weather["wind"]
        avg_baro = weather["pressure"]

    st.markdown("---")
    predict_btn = st.button("🔍  Run Flood Risk Analysis", use_container_width=True)

# ══════════════ MAIN ══════════════
st.markdown(f'<div class="hero-title">🌊 FloodWatch Chennai <span class="live-badge">● LIVE</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">CNN-LSTM Deep Learning · SHAP Explainability · Monte Carlo Uncertainty · Live Weather via Open-Meteo · Real IMD Data 1993–2023</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Flood Risk Prediction", "🌤️ Live Weather", "📊 7-Day Forecast", "ℹ️ Model Info"])

# ── TAB 1: Prediction ──
with tab1:
    model  = load_flood_model()
    scaler = get_scaler()

    if model is None:
        st.error("⚠️ `cnn_lstm_chennai_flood.h5` not found in the repo. Upload it to GitHub.")
        st.stop()

    month_num = datetime.now().month
    is_ne = 1 if month_num in [10,11,12] else 0
    is_sw = 1 if month_num in [6,7,8,9]  else 0
    avg_rain = total_rain / max(rainy_days, 1)

    input_vec = np.array([[
        total_rain, max_rain, rainy_days, avg_rain,
        avg_temp, avg_hum, avg_wind, avg_baro,
        1.43, 1.20, is_ne, is_sw, lag1, lag2
    ]])
    X_sc = scaler.transform(input_vec)
    X_3d = X_sc.reshape(1, 1, len(FEATURES))

    # Auto-predict on live mode, button on manual
    run_now = predict_btn or (data_mode == "🌐 Live (Open-Meteo API)")

    if run_now:
        with st.spinner("Running Monte Carlo Dropout (50 passes)..."):
            prob_mean, prob_std = mc_predict(model, X_3d)
            zname, zbadge, zcol, zemoji = risk_zone(prob_mean)
            clabel, ccolor = conf_label(prob_std)

        # ── Live data banner ──
        if weather["success"] and data_mode == "🌐 Live (Open-Meteo API)":
            st.markdown(f'<div class="info-box">🌐 <b>Live data fetched</b> from Open-Meteo API for {zone_name} at {weather["fetched_at"]} &nbsp;|&nbsp; Current rainfall: {weather["precip_now"]} mm/hr</div>', unsafe_allow_html=True)

        # ── Result cards ──
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Flood Probability</div><div class="metric-value {zcol}">{prob_mean*100:.1f}%</div><div style="font-size:0.75rem;color:#475569;margin-top:0.3rem;">MC mean · 50 passes</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Risk Zone</div><div class="metric-value {zcol}">{zemoji} {zname}</div><div style="margin-top:0.4rem;"><span class="{zbadge}">{zname}</span></div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Model Confidence</div><div class="metric-value" style="color:{ccolor};">{clabel}</div><div style="font-size:0.75rem;color:#475569;margin-top:0.3rem;">σ = {prob_std:.4f}</div></div>', unsafe_allow_html=True)
        with c4:
            season = "⛈️ NE Monsoon" if is_ne else ("🌧️ SW Monsoon" if is_sw else "☀️ Dry Season")
            st.markdown(f'<div class="metric-card"><div class="metric-label">Current Season</div><div class="metric-value" style="color:#38bdf8;font-size:1rem;margin-top:0.5rem;">{season}</div><div style="font-size:0.75rem;color:#475569;margin-top:0.3rem;">{datetime.now().strftime("%B %Y")}</div></div>', unsafe_allow_html=True)

        # ── Advisory ──
        if zname == "Very High":
            st.markdown(f'<div class="danger-box">🔴 <b>VERY HIGH FLOOD RISK — {zone_name}</b><br>Critical susceptibility detected. Authorities should activate emergency protocols. Residents near Adyar River, Cooum, Buckingham Canal must be on high alert.</div>', unsafe_allow_html=True)
        elif zname == "High":
            st.markdown(f'<div class="warn-box">🟠 <b>HIGH FLOOD RISK — {zone_name}</b><br>Elevated susceptibility. Pre-position emergency resources. Monitor NDRF advisories. Check storm drains in low-lying areas.</div>', unsafe_allow_html=True)
        elif zname == "Medium":
            st.markdown(f'<div class="info-box">🟡 <b>MEDIUM RISK — {zone_name}</b><br>Moderate probability of waterlogging. Stay alert for IMD heavy rainfall warnings. Avoid underpasses during heavy rain.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="safe-box">🟢 <b>LOW RISK — {zone_name}</b><br>Current conditions show low flood susceptibility. Standard monitoring recommended. Normal activities can continue.</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── SHAP + MC side by side ──
        col_shap, col_mc = st.columns([3, 2])

        with col_shap:
            st.markdown('<div class="section-header">🔬 SHAP Explainability — Why this prediction?</div>', unsafe_allow_html=True)
            with st.spinner("Computing SHAP values..."):
                try:
                    bg = np.tile(X_sc, (10,1))
                    def mfn(x):
                        return model.predict(x.reshape(x.shape[0],1,x.shape[1]), verbose=0)
                    ex = shap.KernelExplainer(mfn, bg)
                    sv = ex.shap_values(X_sc, nsamples=80)
                    shap_vals = sv[0][0] if isinstance(sv, list) else sv[0]

                    feat_labels = ['Total Rainfall','Max Daily Rain','Rainy Days','Avg Rainfall',
                                   'Temperature','Humidity','Wind Speed','Pressure',
                                   'Inundation Risk','Stagnation Risk',
                                   'NE Monsoon','SW Monsoon','Lag1 Rain','Lag2 Rain']
                    shap_df = pd.DataFrame({'Feature':feat_labels,'SHAP':shap_vals}).reindex(
                        pd.Series(shap_vals).abs().sort_values().index).tail(8)

                    fig, ax = plt.subplots(figsize=(7,3.5))
                    fig.patch.set_facecolor('#111827'); ax.set_facecolor('#111827')
                    colors = ['#ef4444' if v>0 else '#3b82f6' for v in shap_df['SHAP']]
                    bars = ax.barh(shap_df['Feature'], shap_df['SHAP'], color=colors, height=0.55)
                    ax.axvline(0, color='#334155', linewidth=1)
                    ax.set_xlabel('SHAP Value  (red=increases risk, blue=decreases risk)', color='#64748b', fontsize=8)
                    ax.set_title(f'Feature contributions → {zname} risk ({prob_mean*100:.1f}%)', color='#e2e8f0', fontsize=9)
                    ax.tick_params(colors='#94a3b8', labelsize=8)
                    for sp in ax.spines.values(): sp.set_color('#1e293b')
                    for bar, val in zip(bars, shap_df['SHAP']):
                        ax.text(val+(0.001 if val>=0 else -0.001), bar.get_y()+bar.get_height()/2,
                                f'{val:+.4f}', va='center', ha='left' if val>=0 else 'right',
                                color='#94a3b8', fontsize=7)
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()

                    top_i = int(np.argmax(np.abs(shap_vals)))
                    direction = "↑ increases" if shap_vals[top_i]>0 else "↓ decreases"
                    st.markdown(f'<div class="info-box">📌 <b>Primary driver:</b> {feat_labels[top_i]} {direction} flood risk most (SHAP = {shap_vals[top_i]:+.4f})</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"SHAP note: {e}")

        with col_mc:
            st.markdown('<div class="section-header">📉 Monte Carlo Uncertainty</div>', unsafe_allow_html=True)
            all_preds = np.array([model(X_3d, training=True).numpy().flatten()[0] for _ in range(50)])
            fig2, ax2 = plt.subplots(figsize=(4.5,3.5))
            fig2.patch.set_facecolor('#111827'); ax2.set_facecolor('#111827')
            ax2.hist(all_preds, bins=18, color='#2563eb', edgecolor='#1e3a5f', alpha=0.85)
            ax2.axvline(prob_mean, color='#f59e0b', linewidth=2, label=f'Mean={prob_mean:.3f}')
            ax2.axvline(prob_mean-prob_std, color='#64748b', linewidth=1, linestyle='--', label=f'σ={prob_std:.4f}')
            ax2.axvline(prob_mean+prob_std, color='#64748b', linewidth=1, linestyle='--')
            ax2.set_xlabel('Flood Probability', color='#64748b', fontsize=8)
            ax2.set_ylabel('Frequency / 50 passes', color='#64748b', fontsize=8)
            ax2.set_title('Dropout uncertainty distribution', color='#e2e8f0', fontsize=9)
            ax2.tick_params(colors='#94a3b8', labelsize=7)
            ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#94a3b8', fontsize=7)
            for sp in ax2.spines.values(): sp.set_color('#1e293b')
            plt.tight_layout(); st.pyplot(fig2); plt.close()

    else:
        st.markdown('<div style="text-align:center;padding:3rem;color:#334155;"><div style="font-size:3.5rem;">🌊</div><div style="font-family:Space Mono,monospace;font-size:1rem;color:#4a7fa5;margin-top:1rem;">Select a zone → click Run Flood Risk Analysis</div></div>', unsafe_allow_html=True)

# ── TAB 2: Live Weather ──
with tab2:
    st.markdown('<div class="section-header">🌤️ Current Live Weather — Chennai (Open-Meteo API)</div>', unsafe_allow_html=True)

    if weather["success"]:
        st.markdown(f'<div class="safe-box">✅ Live data successfully fetched at {weather["fetched_at"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn-box">⚠️ API unavailable: {weather.get("error","")} — showing fallback values</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    metrics = [
        (c1,"🌡️ Temperature",    f'{weather["temp"]}°C',         "Live from Open-Meteo"),
        (c2,"💧 Humidity",        f'{weather["humidity"]}%',      "Relative humidity"),
        (c3,"💨 Wind Speed",      f'{weather["wind"]} km/h',      "10m above ground"),
        (c4,"📊 Pressure",        f'{weather["pressure"]} hPa',   "Surface pressure"),
    ]
    for col,label,val,sub in metrics:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="color:#38bdf8;font-size:1.6rem;">{val}</div><div style="font-size:0.75rem;color:#475569;margin-top:0.3rem;">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">🌧️ Current Rainfall</div><div class="metric-value" style="color:#60a5fa;">{weather["precip_now"]} mm/hr</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">📅 30-Day Total Rainfall</div><div class="metric-value" style="color:#60a5fa;">{weather["total_rain"]} mm</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">🌦️ Rainy Days (30-day)</div><div class="metric-value" style="color:#60a5fa;">{weather["rainy_days"]} days</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="info-box">📡 <b>Data source:</b> Open-Meteo.com — free, open-source weather API. No API key required. Data from ECMWF, NOAA, DWD models. Updated every 30 minutes in this app. Attribution: CC BY 4.0</div>', unsafe_allow_html=True)

# ── TAB 3: 7-Day Forecast ──
with tab3:
    st.markdown('<div class="section-header">📅 7-Day Flood Risk Forecast — Chennai</div>', unsafe_allow_html=True)

    if weather["forecast_dates"] and model and scaler:
        forecast_dates = weather["forecast_dates"][:7]
        forecast_rain  = weather["forecast_rain"][:7]
        forecast_prob  = weather["forecast_prob"][:7]

        month_num_now = datetime.now().month
        is_ne_now = 1 if month_num_now in [10,11,12] else 0
        is_sw_now = 1 if month_num_now in [6,7,8,9]  else 0

        risk_results = []
        for i, (date, daily_r, prob_r) in enumerate(zip(forecast_dates, forecast_rain, forecast_prob)):
            # Scale daily rain → monthly equivalent estimate
            est_monthly = daily_r * 20 + total_rain * 0.6
            est_max     = daily_r * 1.5
            est_days    = max(1, int(daily_r > 1) * 8)
            iv = np.array([[est_monthly, est_max, est_days, daily_r,
                            avg_temp, avg_hum, avg_wind, avg_baro,
                            1.43, 1.20, is_ne_now, is_sw_now,
                            total_rain, est_monthly * 0.8]])
            Xs = scaler.transform(iv).reshape(1,1,len(FEATURES))
            pm, ps = mc_predict(model, Xs, n=20)
            zn, zb, zc, ze = risk_zone(pm)
            risk_results.append({"date":date,"rain":daily_r,"prob":pm,"sigma":ps,
                                  "zone":zn,"badge":zb,"color":zc,"emoji":ze,
                                  "precip_prob":prob_r})

        # Table view
        cols = st.columns(7)
        for col, r in zip(cols, risk_results):
            dt = datetime.strptime(r["date"],"%Y-%m-%d")
            day_str = dt.strftime("%a\n%d %b")
            with col:
                st.markdown(f"""
                <div class="metric-card" style="text-align:center;padding:0.8rem 0.4rem;">
                    <div style="font-family:Space Mono,monospace;font-size:0.72rem;color:#64748b;">{dt.strftime('%a')}</div>
                    <div style="font-family:Space Mono,monospace;font-size:0.8rem;color:#94a3b8;">{dt.strftime('%d %b')}</div>
                    <div style="font-size:1.6rem;margin:0.4rem 0;">{r['emoji']}</div>
                    <div class="{r['badge']}" style="font-size:0.62rem;">{r['zone']}</div>
                    <div style="font-size:0.75rem;color:#60a5fa;margin-top:0.4rem;">{r['rain']:.1f} mm</div>
                    <div style="font-size:0.72rem;color:#475569;">{r['prob']*100:.0f}% risk</div>
                </div>""", unsafe_allow_html=True)

        # Bar chart
        st.markdown("")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5))
        fig.patch.set_facecolor('#111827')
        for ax in [ax1, ax2]: ax.set_facecolor('#111827')

        dates_short = [datetime.strptime(r["date"],"%Y-%m-%d").strftime("%d %b") for r in risk_results]
        probs = [r["prob"]*100 for r in risk_results]
        rains = [r["rain"] for r in risk_results]
        bar_colors = ['#ef4444' if p>=75 else '#f97316' if p>=50 else '#f59e0b' if p>=25 else '#22c55e' for p in probs]

        ax1.bar(dates_short, probs, color=bar_colors, edgecolor='#0a0f1e', linewidth=0.5)
        ax1.axhline(75, color='#ef4444', linestyle='--', linewidth=0.8, alpha=0.6, label='Very High threshold')
        ax1.axhline(50, color='#f97316', linestyle='--', linewidth=0.8, alpha=0.6, label='High threshold')
        ax1.set_ylabel('Flood Risk %', color='#94a3b8', fontsize=9)
        ax1.set_title('7-Day CNN-LSTM Flood Risk Forecast — Chennai', color='#e2e8f0', fontsize=10)
        ax1.tick_params(colors='#94a3b8', labelsize=8); ax1.set_ylim([0,100])
        ax1.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#94a3b8', fontsize=7)
        for sp in ax1.spines.values(): sp.set_color('#1e293b')

        ax2.bar(dates_short, rains, color='#3b82f6', edgecolor='#0a0f1e', linewidth=0.5)
        ax2.set_ylabel('Rainfall (mm)', color='#94a3b8', fontsize=9)
        ax2.set_title('Forecast Daily Rainfall', color='#e2e8f0', fontsize=10)
        ax2.tick_params(colors='#94a3b8', labelsize=8)
        for sp in ax2.spines.values(): sp.set_color('#1e293b')

        plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("7-day forecast requires live API connection and model. Please ensure model is loaded.")

# ── TAB 4: Model Info ──
with tab4:
    st.markdown('<div class="section-header">🧠 Model Architecture & Performance</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="metric-card">
        <div class="metric-label">CNN-LSTM Architecture</div>
        <div style="font-family:Space Mono,monospace;font-size:0.85rem;color:#38bdf8;line-height:2.2;">
            Input (1 × 14 features)<br>↓ Conv1D (64 filters, k=1, ReLU)<br>
            ↓ Dropout (0.3) ← Monte Carlo<br>↓ LSTM (64 units)<br>
            ↓ Dropout (0.3) ← Monte Carlo<br>↓ Dense (32, ReLU)<br>
            ↓ Dropout (0.2)<br>↓ Dense (1, Sigmoid)<br>Output: Flood Probability [0–1]
        </div></div>""", unsafe_allow_html=True)
    with c2:
        stats = [("Accuracy","84%"),("AUC-ROC","0.97"),("Training data","1993–2023 (30 years)"),
                 ("Stations","62 IMD stations Chennai"),("MC passes","50 forward passes"),
                 ("Risk levels","4 (Low/Medium/High/Very High)"),("Optimizer","Adam"),
                 ("Base paper","Razavi-Termeh et al., Springer 2025")]
        st.markdown('<div class="metric-card"><div class="metric-label">Performance Stats</div>', unsafe_allow_html=True)
        for k,v in stats:
            st.markdown(f'<div class="weather-stat"><span>{k}</span><span>{v}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-header">📦 Real Datasets Used</div>', unsafe_allow_html=True)
    datasets = [
        ("🌧️","Chennai Rainfall 1993–2023","IMD","21,416 daily records · 62 stations"),
        ("🗺️","Chennai 2015 Inundation Zones","Chennai GCC","4,001 grid points · real 2015 mega-flood"),
        ("💧","Stagnation Points 2015","Chennai GCC","Real waterlogging locations · all zones"),
        ("🌡️","Chennai Weather 2009–2024","IMD","Temperature · Humidity · Wind · Pressure"),
        ("🌐","Live Weather Feed","Open-Meteo API","Real-time · No API key · Updates every 30 min"),
    ]
    for icon,name,src,desc in datasets:
        st.markdown(f'<div class="metric-card" style="display:flex;gap:1rem;align-items:center;"><div style="font-size:1.6rem;">{icon}</div><div><div style="font-family:Space Mono,monospace;font-size:0.82rem;color:#e2e8f0;">{name}</div><div style="font-size:0.75rem;color:#3b82f6;">Source: {src}</div><div style="font-size:0.78rem;color:#64748b;">{desc}</div></div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="metric-card"><div class="metric-label">Project Details</div><div style="font-family:Space Mono,monospace;font-size:0.85rem;color:#e2e8f0;line-height:1.8;">Flood Susceptibility Mapping for South Tamil Nadu<br>using Optimized CNN-LSTM with Uncertainty Quantification<br>and Explainable Risk Zoning</div><div style="font-size:0.8rem;color:#64748b;margin-top:0.5rem;">M.Tech AI & DS · B.S. Abdur Rahman Crescent Institute · 2025–2027</div></div>', unsafe_allow_html=True)
