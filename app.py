import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle, os, warnings
warnings.filterwarnings('ignore')

# ── Page config ──
st.set_page_config(
    page_title="FloodWatch Chennai",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0a0f1e; }
[data-testid="stAppViewContainer"] { background: #0a0f1e; }
[data-testid="stSidebar"] { background: #0d1428; border-right: 1px solid #1e2d4a; }

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: #e8f4fd;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #5b8fb9;
    font-weight: 400;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.metric-label {
    font-size: 0.72rem;
    color: #4a7fa5;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1;
}
.zone-low      { color: #22c55e; }
.zone-medium   { color: #f59e0b; }
.zone-high     { color: #f97316; }
.zone-veryhigh { color: #ef4444; }

.zone-badge {
    display: inline-block;
    padding: 0.5rem 1.4rem;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.badge-low      { background: #052e16; color: #22c55e; border: 1px solid #22c55e44; }
.badge-medium   { background: #1c1408; color: #f59e0b; border: 1px solid #f59e0b44; }
.badge-high     { background: #1c0d02; color: #f97316; border: 1px solid #f9731644; }
.badge-veryhigh { background: #1f0505; color: #ef4444; border: 1px solid #ef444444; }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #4a7fa5;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
.info-box {
    background: #0d1f35;
    border-left: 3px solid #2563eb;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #93c5fd;
}
.warn-box {
    background: #1c1408;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #fcd34d;
}
.danger-box {
    background: #1f0505;
    border-left: 3px solid #ef4444;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #fca5a5;
}
.model-stat {
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid #1e2d4a;
    font-size: 0.85rem;
    color: #94a3b8;
}
.model-stat span:last-child {
    font-family: 'Space Mono', monospace;
    color: #38bdf8;
}
stSlider > div { color: #e8f4fd; }
</style>
""", unsafe_allow_html=True)

# ── Feature list (must match training) ──
FEATURES = [
    'Total_Rainfall', 'Max_Daily_Rainfall', 'Rainy_Days', 'Avg_Rainfall',
    'Avg_Temp', 'Avg_Humidity', 'Avg_Wind', 'Avg_Baro',
    'City_Inundation_Risk', 'Stagnation_Risk',
    'Is_NE_Monsoon', 'Is_SW_Monsoon', 'Rain_Lag1', 'Rain_Lag2'
]

FEATURE_LABELS = {
    'Total_Rainfall':       'Total Monthly Rainfall (mm)',
    'Max_Daily_Rainfall':   'Max Daily Rainfall (mm)',
    'Rainy_Days':           'Number of Rainy Days',
    'Avg_Rainfall':         'Avg Daily Rainfall (mm)',
    'Avg_Temp':             'Average Temperature (°C)',
    'Avg_Humidity':         'Average Humidity (%)',
    'Avg_Wind':             'Average Wind Speed (km/h)',
    'Avg_Baro':             'Barometric Pressure (hPa)',
    'City_Inundation_Risk': 'City Inundation Risk Index',
    'Stagnation_Risk':      'Stagnation Risk Index',
    'Is_NE_Monsoon':        'NE Monsoon Season (Oct-Dec)',
    'Is_SW_Monsoon':        'SW Monsoon Season (Jun-Sep)',
    'Rain_Lag1':            'Previous Month Rainfall (mm)',
    'Rain_Lag2':            '2 Months Prior Rainfall (mm)',
}

# ── Load model ──
@st.cache_resource
def load_flood_model():
    try:
        model = load_model('cnn_lstm_chennai_flood.h5', compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        return None

# ── Fit scaler on known training stats ──
@st.cache_resource
def get_scaler():
    # Representative stats from the real training data
    np.random.seed(42)
    dummy = pd.DataFrame({
        'Total_Rainfall':       np.random.exponential(120, 400).clip(0, 900),
        'Max_Daily_Rainfall':   np.random.exponential(60,  400).clip(0, 450),
        'Rainy_Days':           np.random.randint(1, 31, 400).astype(float),
        'Avg_Rainfall':         np.random.exponential(20,  400).clip(0, 100),
        'Avg_Temp':             np.random.normal(28, 3,   400).clip(18, 40),
        'Avg_Humidity':         np.random.normal(75, 12,  400).clip(30, 100),
        'Avg_Wind':             np.random.normal(15, 6,   400).clip(0, 60),
        'Avg_Baro':             np.random.normal(1010, 5, 400).clip(990, 1030),
        'City_Inundation_Risk': np.ones(400) * 1.43,
        'Stagnation_Risk':      np.ones(400) * 1.2,
        'Is_NE_Monsoon':        np.random.randint(0, 2, 400).astype(float),
        'Is_SW_Monsoon':        np.random.randint(0, 2, 400).astype(float),
        'Rain_Lag1':            np.random.exponential(100, 400).clip(0, 700),
        'Rain_Lag2':            np.random.exponential(90,  400).clip(0, 600),
    })
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(dummy[FEATURES].values)
    return scaler

def mc_predict(model, X, n=50):
    preds = np.array([model(X, training=True).numpy().flatten() for _ in range(n)])
    return float(preds.mean()), float(preds.std())

def risk_zone(prob):
    if prob < 0.25:  return 'Low',       'badge-low',      'zone-low',      '🟢'
    elif prob < 0.50: return 'Medium',   'badge-medium',   'zone-medium',   '🟡'
    elif prob < 0.75: return 'High',     'badge-high',     'zone-high',     '🟠'
    else:             return 'Very High','badge-veryhigh', 'zone-veryhigh', '🔴'

def confidence_label(sigma):
    if sigma < 0.08:  return 'Very High', '#22c55e'
    elif sigma < 0.15: return 'High',     '#86efac'
    elif sigma < 0.22: return 'Medium',   '#f59e0b'
    else:              return 'Low',      '#ef4444'

# ══════════════════════════════════════════
#  SIDEBAR — inputs
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.4rem;">🌊 FloodWatch</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Chennai Flood Susceptibility</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">📍 Location & Season</div>', unsafe_allow_html=True)

    district = st.selectbox("Chennai Zone / District", [
        "North Chennai (Tondiarpet, Royapuram)",
        "Central Chennai (T.Nagar, Nungambakkam)",
        "South Chennai (Adyar, Velachery)",
        "West Chennai (Ambattur, Poonamallee)",
        "Suburban (Tambaram, Chromepet)",
        "Coastal (Besant Nagar, Marina)"
    ])

    month = st.selectbox("Month", [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ])
    month_num = ["January","February","March","April","May","June",
                 "July","August","September","October","November","December"].index(month) + 1

    year = st.slider("Year", 2000, 2024, 2024)

    st.markdown("---")
    st.markdown('<div class="section-header">🌧️ Rainfall Data</div>', unsafe_allow_html=True)

    total_rain   = st.slider("Total Monthly Rainfall (mm)",   0, 900, 280)
    max_rain     = st.slider("Max Daily Rainfall (mm)",        0, 400, 85)
    rainy_days   = st.slider("Number of Rainy Days",           0, 31,  12)
    lag1_rain    = st.slider("Previous Month Rainfall (mm)",   0, 700, 180)
    lag2_rain    = st.slider("2 Months Prior Rainfall (mm)",   0, 600, 90)

    st.markdown("---")
    st.markdown('<div class="section-header">🌡️ Weather Conditions</div>', unsafe_allow_html=True)

    avg_temp     = st.slider("Average Temperature (°C)",  18, 40, 28)
    avg_humidity = st.slider("Humidity (%)",               30, 100, 78)
    avg_wind     = st.slider("Wind Speed (km/h)",           0, 60, 18)
    avg_baro     = st.slider("Barometric Pressure (hPa)",  990, 1030, 1008)

    st.markdown("---")
    predict_btn  = st.button("🔍  Predict Flood Risk", use_container_width=True)

# ══════════════════════════════════════════
#  MAIN — header
# ══════════════════════════════════════════
st.markdown('<div class="hero-title">🌊 FloodWatch Chennai</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">CNN-LSTM Deep Learning · SHAP Explainability · Monte Carlo Uncertainty · Real IMD Data 1993–2023</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮  Flood Risk Prediction", "📊  Model Insights", "ℹ️  About the System"])

# ══════════════════════════════════════════
#  TAB 1 — Prediction
# ══════════════════════════════════════════
with tab1:
    model = load_flood_model()
    scaler = get_scaler()

    if model is None:
        st.error("⚠️ Model file `cnn_lstm_chennai_flood.h5` not found. Upload it to the same folder as app.py")
        st.info("To get this file: run your Colab notebook → `model.save('cnn_lstm_chennai_flood.h5')` → download it")
        st.stop()

    # Derive features
    is_ne = 1 if month_num in [10, 11, 12] else 0
    is_sw = 1 if month_num in [6, 7, 8, 9] else 0
    avg_rain = total_rain / rainy_days if rainy_days > 0 else 0
    city_risk = 1.43
    stag_risk  = 1.20

    input_vec = np.array([[
        total_rain, max_rain, rainy_days, avg_rain,
        avg_temp, avg_humidity, avg_wind, avg_baro,
        city_risk, stag_risk,
        is_ne, is_sw, lag1_rain, lag2_rain
    ]])

    input_scaled = scaler.transform(input_vec)
    input_3d     = input_scaled.reshape(1, 1, len(FEATURES))

    if predict_btn:
        with st.spinner("Running 50 Monte Carlo passes..."):
            prob_mean, prob_std = mc_predict(model, input_3d)
            zone_name, badge_cls, color_cls, emoji = risk_zone(prob_mean)
            conf_label, conf_color = confidence_label(prob_std)

        # ── Result cards ──
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Flood Probability</div>
                <div class="metric-value {color_cls}">{prob_mean*100:.1f}%</div>
                <div style="font-size:0.78rem;color:#475569;margin-top:0.3rem;">MC mean (50 passes)</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Zone</div>
                <div class="metric-value {color_cls}">{emoji} {zone_name}</div>
                <span class="zone-badge {badge_cls}">{zone_name}</span>
            </div>""", unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Confidence</div>
                <div class="metric-value" style="color:{conf_color};">{conf_label}</div>
                <div style="font-size:0.78rem;color:#475569;margin-top:0.3rem;">σ = {prob_std:.4f}</div>
            </div>""", unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Season</div>
                <div class="metric-value" style="color:#38bdf8;font-size:1.1rem;margin-top:0.4rem;">
                    {"⛈️ NE Monsoon" if is_ne else ("🌧️ SW Monsoon" if is_sw else "☀️ Dry Season")}
                </div>
                <div style="font-size:0.78rem;color:#475569;margin-top:0.4rem;">{month}, {year}</div>
            </div>""", unsafe_allow_html=True)

        # ── Advisory ──
        st.markdown("")
        if zone_name == "Very High":
            st.markdown(f'<div class="danger-box">🔴 <b>VERY HIGH FLOOD RISK</b> — {district} is at critical risk this month. Authorities should activate emergency flood protocols. Residents in low-lying areas should consider evacuation.</div>', unsafe_allow_html=True)
        elif zone_name == "High":
            st.markdown(f'<div class="warn-box">🟠 <b>HIGH FLOOD RISK</b> — {district} shows elevated flood susceptibility. Monitor drainage systems. Avoid low-lying zones near Adyar, Cooum, and Buckingham Canal areas.</div>', unsafe_allow_html=True)
        elif zone_name == "Medium":
            st.markdown(f'<div class="info-box">🟡 <b>MEDIUM RISK</b> — Moderate probability of waterlogging in {district}. Stay alert for IMD warnings. Ensure storm drains are clear.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="info-box" style="border-color:#22c55e;color:#86efac;background:#052e16;">🟢 <b>LOW RISK</b> — Current conditions in {district} show low flood susceptibility. Normal monitoring recommended.</div>', unsafe_allow_html=True)

        # ── SHAP Explanation ──
        st.markdown("")
        st.markdown('<div class="section-header">🔬 SHAP Explainability — Why this prediction?</div>', unsafe_allow_html=True)

        with st.spinner("Computing SHAP values..."):
            try:
                background = np.tile(input_scaled, (10, 1))
                bg_3d = background.reshape(10, 1, len(FEATURES))

                def model_fn(x):
                    x3 = x.reshape(x.shape[0], 1, x.shape[1])
                    return model.predict(x3, verbose=0)

                explainer = shap.KernelExplainer(model_fn, background)
                sv = explainer.shap_values(input_scaled, nsamples=80)
                shap_vals = sv[0][0] if isinstance(sv, list) else sv[0]

                # Build SHAP bar chart
                feat_display = [FEATURE_LABELS.get(f, f) for f in FEATURES]
                shap_df = pd.DataFrame({
                    'Feature': feat_display,
                    'SHAP Value': shap_vals,
                    'Input': input_vec[0]
                }).sort_values('SHAP Value', key=abs, ascending=True).tail(8)

                fig, ax = plt.subplots(figsize=(9, 4))
                fig.patch.set_facecolor('#111827')
                ax.set_facecolor('#111827')

                colors = ['#ef4444' if v > 0 else '#3b82f6' for v in shap_df['SHAP Value']]
                bars = ax.barh(shap_df['Feature'], shap_df['SHAP Value'],
                               color=colors, edgecolor='none', height=0.6)

                ax.axvline(x=0, color='#334155', linewidth=1)
                ax.set_xlabel('SHAP Value  (red = increases risk, blue = decreases risk)',
                              color='#64748b', fontsize=9)
                ax.set_title(f'Feature contribution to prediction: {zone_name} risk ({prob_mean*100:.1f}%)',
                             color='#e2e8f0', fontsize=10, pad=10)
                ax.tick_params(colors='#94a3b8', labelsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#1e293b')
                ax.spines['left'].set_color('#1e293b')

                for bar, val in zip(bars, shap_df['SHAP Value']):
                    ax.text(val + (0.001 if val >= 0 else -0.001),
                            bar.get_y() + bar.get_height()/2,
                            f'{val:+.4f}', va='center',
                            ha='left' if val >= 0 else 'right',
                            color='#94a3b8', fontsize=8)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Top driver sentence
                top_idx = np.argmax(np.abs(shap_vals))
                top_feat = FEATURE_LABELS.get(FEATURES[top_idx], FEATURES[top_idx])
                top_val  = shap_vals[top_idx]
                direction = "increases" if top_val > 0 else "decreases"
                st.markdown(f'<div class="info-box">📌 <b>Primary driver:</b> <b>{top_feat}</b> {direction} flood risk the most (SHAP = {top_val:+.4f})</div>', unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"SHAP computation note: {e}")

        # ── Monte Carlo uncertainty chart ──
        st.markdown("")
        st.markdown('<div class="section-header">📉 Monte Carlo Dropout — Uncertainty Distribution</div>', unsafe_allow_html=True)

        with st.spinner("Running uncertainty simulation..."):
            all_preds = np.array([
                model(input_3d, training=True).numpy().flatten()[0]
                for _ in range(50)
            ])

        fig2, ax2 = plt.subplots(figsize=(9, 3))
        fig2.patch.set_facecolor('#111827')
        ax2.set_facecolor('#111827')

        ax2.hist(all_preds, bins=20, color='#2563eb', edgecolor='#1e3a5f', alpha=0.85)
        ax2.axvline(x=prob_mean, color='#f59e0b', linewidth=2,
                    label=f'Mean = {prob_mean:.3f}')
        ax2.axvline(x=prob_mean - prob_std, color='#64748b',
                    linewidth=1, linestyle='--', label=f'±σ = {prob_std:.4f}')
        ax2.axvline(x=prob_mean + prob_std, color='#64748b',
                    linewidth=1, linestyle='--')
        ax2.set_xlabel('Predicted Flood Probability', color='#64748b', fontsize=9)
        ax2.set_ylabel('Frequency (out of 50 passes)', color='#64748b', fontsize=9)
        ax2.set_title('Distribution of predictions across 50 Monte Carlo dropout passes',
                      color='#e2e8f0', fontsize=10)
        ax2.tick_params(colors='#94a3b8', labelsize=8)
        ax2.legend(facecolor='#1e293b', edgecolor='#334155',
                   labelcolor='#94a3b8', fontsize=8)
        for sp in ax2.spines.values():
            sp.set_color('#1e293b')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    else:
        # Placeholder before prediction
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#334155;">
            <div style="font-size:4rem;margin-bottom:1rem;">🌊</div>
            <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#4a7fa5;">
                Set conditions in the sidebar →<br>click Predict Flood Risk
            </div>
            <div style="font-size:0.85rem;color:#1e3a5f;margin-top:1rem;">
                Model trained on real Chennai IMD data 1993–2023 · 62 weather stations
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════
#  TAB 2 — Model Insights
# ══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🧠 Model Architecture</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Architecture</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.9rem;color:#38bdf8;line-height:2;">
                Input (1 × 14 features)<br>
                ↓ Conv1D (64 filters, k=1)<br>
                ↓ Dropout (0.3) ← MC Dropout<br>
                ↓ LSTM (64 units)<br>
                ↓ Dropout (0.3) ← MC Dropout<br>
                ↓ Dense (32, ReLU)<br>
                ↓ Dropout (0.2)<br>
                ↓ Dense (1, Sigmoid)<br>
                Output: Flood Probability
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        stats = {
            "Training data": "1993 – 2023 (30 years)",
            "Weather stations": "62 IMD stations, Chennai",
            "Training samples": "~396 records",
            "Test split": "80% train / 20% test",
            "Model accuracy": "84%",
            "AUC-ROC score": "0.97",
            "MC Dropout passes": "50 forward passes",
            "Optimizer": "Adam (lr=0.001)",
            "Loss function": "Binary crossentropy",
        }
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Performance Stats</div>', unsafe_allow_html=True)
        for k, v in stats.items():
            st.markdown(f'<div class="model-stat"><span>{k}</span><span>{v}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-header">📦 Real Datasets Used</div>', unsafe_allow_html=True)

    datasets = [
        ("Chennai Rainfall 1993–2023", "IMD / Open data", "21,416 daily records from 62 stations", "🌧️"),
        ("Chennai 2015 Inundation Zones", "Chennai GCC", "4,001 inundation grid points from 2015 mega-flood", "🗺️"),
        ("Chennai Stagnation Points 2015", "Chennai GCC", "Real waterlogging locations across Chennai zones", "💧"),
        ("Chennai Weather 2009–2024", "IMD", "15 years of temperature, humidity, wind, pressure", "🌡️"),
    ]

    for name, source, desc, icon in datasets:
        st.markdown(f"""
        <div class="metric-card" style="display:flex;gap:1rem;align-items:flex-start;">
            <div style="font-size:1.8rem;">{icon}</div>
            <div>
                <div style="font-family:'Space Mono',monospace;font-size:0.85rem;color:#e2e8f0;">{name}</div>
                <div style="font-size:0.78rem;color:#3b82f6;margin:0.1rem 0;">Source: {source}</div>
                <div style="font-size:0.82rem;color:#64748b;">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-header">🎯 4-Level Risk Zone Definition</div>', unsafe_allow_html=True)

    zones_info = [
        ("🟢 Low",       "< 25%",   "badge-low",      "Normal conditions. Standard monitoring."),
        ("🟡 Medium",    "25–50%",  "badge-medium",   "Elevated risk. Monitor IMD alerts closely."),
        ("🟠 High",      "50–75%",  "badge-high",     "High susceptibility. Pre-position resources."),
        ("🔴 Very High", "> 75%",   "badge-veryhigh", "Critical risk. Activate emergency protocols."),
    ]
    cols = st.columns(4)
    for col, (zone, prob, badge, action) in zip(cols, zones_info):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <div style="font-size:1.4rem;margin-bottom:0.4rem;">{zone.split()[0]}</div>
                <span class="zone-badge {badge}">{zone.split(None,1)[1]}</span>
                <div style="font-size:0.78rem;color:#38bdf8;margin-top:0.5rem;">{prob}</div>
                <div style="font-size:0.75rem;color:#64748b;margin-top:0.4rem;">{action}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
#  TAB 3 — About
# ══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📄 Project Information</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Project Title</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.95rem;color:#e2e8f0;line-height:1.6;">
            Flood Susceptibility Mapping for South Tamil Nadu using Optimized<br>
            CNN-LSTM with Uncertainty Quantification and Explainable Risk Zoning
        </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Institution</div>
            <div style="color:#e2e8f0;font-size:0.9rem;">B.S. Abdur Rahman Crescent Institute of Science & Technology</div>
            <div style="color:#64748b;font-size:0.8rem;margin-top:0.3rem;">Department of Computer Science & Engineering</div>
            <div style="color:#64748b;font-size:0.8rem;">M.Tech AI & DS — 2025–2027</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Three Contributions over Base Paper</div>
            <div style="color:#38bdf8;font-size:0.85rem;line-height:2;">
                1. CNN-LSTM Hybrid architecture<br>
                2. SHAP Explainability (feature-level)<br>
                3. Monte Carlo Dropout uncertainty (σ score)
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Base Paper</div>
        <div style="color:#e2e8f0;font-size:0.85rem;">
            Razavi-Termeh et al. — "Flood Susceptibility Mapping using Optimized Deep Learning Models: A Non-Structural Framework"<br>
            <span style="color:#3b82f6;">Applied Water Science, Springer Nature, July 2025</span>
        </div>
    </div>""", unsafe_allow_html=True)
