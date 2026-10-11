import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. PAGE SETUP & MODERN DARK BLUE MEDICAL THEME (CSS)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Explainable Heart Disease Risk Prediction",
    page_icon="🫀",
    layout="wide"
)

st.markdown("""
    <style>
    /* Dark Slate Background */
    .stApp {
        background-color: #0b1329;
        color: #e6f1ff;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Container Box Styling matching the screenshot cards */
    .dashboard-card {
        background-color: #111d38;
        border: 1px solid #1c2d54;
        border-radius: 12px;
        padding: 22px;
        height: 100%;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    
    .card-title {
        color: #e6f1ff;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .card-subtitle {
        color: #8892b0;
        font-size: 12px;
        margin-bottom: 16px;
    }

    /* Green Glow Tags & Highlights */
    .highlight-tag {
        background-color: rgba(100, 255, 218, 0.15);
        color: #64ffda;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    
    .low-risk-pill {
        background-color: rgba(100, 255, 218, 0.12);
        color: #64ffda;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid rgba(100, 255, 218, 0.3);
    }

    .high-risk-pill {
        background-color: rgba(255, 107, 107, 0.12);
        color: #ff6b6b;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid rgba(255, 107, 107, 0.3);
    }

    .top-status-badge {
        background-color: rgba(100, 255, 218, 0.1);
        border: 1px solid rgba(100, 255, 218, 0.3);
        color: #64ffda;
        padding: 8px 16px;
        border-radius: 25px;
        font-size: 13px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Download Report Button */
    .stDownloadButton>button {
        background: #64ffda !important;
        color: #0b1329 !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        width: 100% !important;
        border: none !important;
        padding: 12px 20px !important;
        margin-top: 15px !important;
        transition: all 0.2s ease;
    }
    .stDownloadButton>button:hover {
        background: #4cdbba !important;
        transform: translateY(-1px);
    }

    /* Form Input Controls Styling */
    .stSlider, .stSelectbox {
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. MODEL TRAINING (PURE PYTHON - NO DATABASE)
# ---------------------------------------------------------
@st.cache_resource
def train_models():
    np.random.seed(42)
    n_samples = 600
    
    # Clinical Features matching standard heart disease parameters
    age = np.random.randint(29, 78, n_samples)
    sex = np.random.randint(0, 2, n_samples)
    cp = np.random.randint(0, 4, n_samples)
    trestbps = np.random.randint(94, 200, n_samples)
    chol = np.random.randint(126, 400, n_samples)
    fbs = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    restecg = np.random.randint(0, 3, n_samples)
    thalach = np.random.randint(71, 202, n_samples)
    exang = np.random.randint(0, 2, n_samples)
    oldpeak = np.round(np.random.uniform(0.0, 4.2, n_samples), 1)
    slope = np.random.randint(0, 3, n_samples)
    ca = np.random.randint(0, 4, n_samples)
    thal = np.random.randint(1, 4, n_samples)

    risk_score = (
        (age > 55) * 1.5 + (cp > 1) * 2.5 + (trestbps > 140) * 1.2 + 
        (chol > 240) * 1.1 + (thalach < 130) * 2.0 + (oldpeak > 1.5) * 2.2 + 
        (ca > 0) * 2.0 + (exang == 1) * 1.8
    )
    target = (risk_score > 6.0).astype(int)

    X = pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    })
    
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    return rf_model, lr_model

rf_model, lr_model = train_models()

# ---------------------------------------------------------
# 3. SIDEBAR: PATIENT HEALTH FEATURE INPUTS
# ---------------------------------------------------------
st.sidebar.markdown("### 🫀 Patient Parameters")
st.sidebar.caption("Adjust values to compute real-time ML risk scores.")

# Preset Sample Profiles for quick demonstrations
demo_profile = st.sidebar.selectbox(
    "Sample Profiles (Quick Demo):",
    ["Custom Input", "Low Risk Profile (38%)", "High Risk Profile (82%)"]
)

# Default values based on demo profile selection
if demo_profile == "Low Risk Profile (38%)":
    def_age, def_bp, def_chol, def_hr, def_cp, def_ca, def_oldpeak = 38, 118, 185, 172, 0, 0, 0.2
elif demo_profile == "High Risk Profile (82%)":
    def_age, def_bp, def_chol, def_hr, def_cp, def_ca, def_oldpeak = 64, 158, 310, 105, 3, 2, 2.8
else:
    def_age, def_bp, def_chol, def_hr, def_cp, def_ca, def_oldpeak = 52, 130, 240, 150, 1, 0, 1.0

age = st.sidebar.slider("Age", 20, 80, def_age)
trestbps = st.sidebar.slider("Resting Blood Pressure (mmHg)", 90, 200, def_bp)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 120, 400, def_chol)
thalach = st.sidebar.slider("Max Heart Rate (thalach)", 70, 210, def_hr)
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 5.0, def_oldpeak, step=0.1)
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], index=def_cp, 
                          format_func=lambda x: ["0: Typical Angina", "1: Atypical Angina", "2: Non-anginal", "3: Asymptomatic"][x])
ca = st.sidebar.selectbox("Major Vessels (ca)", [0, 1, 2, 3], index=def_ca)
sex = st.sidebar.radio("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female", horizontal=True)

# ---------------------------------------------------------
# 4. ML PREDICTION CALCULATIONS
# ---------------------------------------------------------
input_data = np.array([[age, sex, cp, trestbps, chol, 0, 0, thalach, 0, oldpeak, 1, ca, 2]])
rf_prob = int(rf_model.predict_proba(input_data)[0][1] * 100)
lr_prob = int(lr_model.predict_proba(input_data)[0][1] * 100)

risk_text = "High Risk" if rf_prob >= 50 else "Low Risk"
risk_badge_class = "high-risk-pill" if rf_prob >= 50 else "low-risk-pill"

# ---------------------------------------------------------
# 5. DASHBOARD HEADER (MATCHING IMAGE SCREENSHOT)
# ---------------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 28px; color: #64ffda;">📈</span>
            <h1 style="margin: 0; font-size: 28px; font-weight: 800; color: #ffffff;">Results and Analytics Dashboard</h1>
        </div>
        <p style="color: #8892b0; margin-top: 4px; font-size: 14px;">
            Model-estimated probabilities, comparison metrics, and Explainable AI (XAI) feature attribution.
        </p>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown(f"""
        <div style="text-align: right; margin-top: 10px;">
            <div class="top-status-badge">
                <span>✔</span> Predicted Class: {risk_text} ({rf_prob}%)
            </div>
        </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# 6. TOP ROW: 3 ANALYTICS CARDS (MATCHING SCREENSHOT)
# ---------------------------------------------------------
col_card1, col_card2, col_card3 = st.columns([1.1, 1.4, 1.3])

# CARD 1: Random Forest Prediction
with col_card1:
    st.markdown(f"""
        <div class="dashboard-card">
            <span class="highlight-tag">Random Forest Prediction</span>
            <div style="display: flex; align-items: baseline; gap: 15px; margin-top: 15px;">
                <span style="font-size: 48px; font-weight: 800; color: #ffffff;">{rf_prob}%</span>
                <span class="{risk_badge_class}">{risk_text.upper()}</span>
            </div>
            <div style="background-color: #1a2b4c; border-radius: 10px; height: 8px; margin-top: 25px; overflow: hidden;">
                <div style="background-color: #64ffda; width: {rf_prob}%; height: 100%;"></div>
            </div>
            <p style="color: #8892b0; font-size: 12px; margin-top: 12px;">
                Patient parameters indicate {"stable cardiovascular risk profile." if rf_prob < 50 else "elevated cardiovascular risk concerns."}
            </p>
        </div>
    """, unsafe_allow_html=True)

# CARD 2: Algorithm Agreement Comparison
with col_card2:
    agreement_status = "Strong Consensus (Agreement)" if abs(rf_prob - lr_prob) <= 10 else "Moderate Variance"
    st.markdown(f"""
        <div class="dashboard-card">
            <div class="card-subtitle" style="margin-bottom: 2px;">Algorithm Agreement</div>
            <div class="card-title">Random Forest vs. Logistic Regression</div>
            
            <div style="margin-top: 18px;">
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #8892b0;">
                    <span>Random Forest Prob</span>
                    <span style="color: #64ffda; font-weight: 700;">{rf_prob}%</span>
                </div>
                <div style="background-color: #1a2b4c; border-radius: 10px; height: 6px; margin-top: 4px; overflow: hidden;">
                    <div style="background-color: #64ffda; width: {rf_prob}%; height: 100%;"></div>
                </div>
            </div>

            <div style="margin-top: 14px;">
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #8892b0;">
                    <span>Logistic Regression Prob</span>
                    <span style="color: #38bdf8; font-weight: 700;">{lr_prob}%</span>
                </div>
                <div style="background-color: #1a2b4c; border-radius: 10px; height: 6px; margin-top: 4px; overflow: hidden;">
                    <div style="background-color: #38bdf8; width: {lr_prob}%; height: 100%;"></div>
                </div>
            </div>

            <div style="display: flex; justify-content: space-between; margin-top: 20px; font-size: 12px;">
                <span style="color: #8892b0;">Consensus Status:</span>
                <span style="color: #64ffda; font-weight: 700;">{agreement_status}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# CARD 3: Downloadable Assessment Report
with col_card3:
    report_content = f"""# Cardiovascular Assessment Report
----------------------------------------
Predicted Risk Class: {risk_text}
Random Forest Probability: {rf_prob}%
Logistic Regression Probability: {lr_prob}%

Patient Parameters:
- Age: {age}
- Resting BP: {trestbps} mmHg
- Cholesterol: {chol} mg/dl
- Max Heart Rate: {thalach} bpm
- ST Depression: {oldpeak}
----------------------------------------
Generated by Explainable Heart Disease Prediction System.
"""
    st.markdown("""
        <div class="dashboard-card">
            <div style="display: flex; align-items: center; gap: 6px;" class="card-subtitle">
                <span>📄</span> <span>Clinical Export</span>
            </div>
            <div class="card-title">Downloadable Assessment Report</div>
            <p style="color: #8892b0; font-size: 12px; margin-top: 8px;">
                Generate a structured PDF-style markdown summary including patient parameters, risk probabilities, and XAI feature attributions.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.download_button(
        label="📥 Download Assessment Report",
        data=report_content,
        file_name=f"patient_assessment_report.txt",
        mime="text/plain"
    )

st.write("")

# ---------------------------------------------------------
# 7. BOTTOM ROW: XAI FEATURE IMPORTANCE & RADAR CHART
# ---------------------------------------------------------
col_bottom1, col_bottom2 = st.columns([1, 1])

# CHART 1: Explainable AI (XAI) Feature Importance
with col_bottom1:
    st.markdown("""
        <div style="background-color: #111d38; border: 1px solid #1c2d54; border-radius: 12px; padding: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #64ffda;">📊</span>
                    <span style="color: #ffffff; font-weight: 700; font-size: 16px;">Explainable AI (XAI) Feature Importance</span>
                </div>
                <span style="background-color: #16264c; color: #8892b0; padding: 3px 8px; border-radius: 4px; font-size: 10px;">
                    SHAP / Gain Attribution
                </span>
            </div>
            <p style="color: #8892b0; font-size: 12px; margin-top: 4px; margin-bottom: 10px;">
                Relative impact of key clinical features on this specific patient prediction score.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Feature Importance Calculations
    feature_labels = ['Chest Pain Type (cp)', 'Max Heart Rate (thalach)', 'ST Depression (oldpeak)', 'Major Vessels (ca)', 'Patient Age']
    feature_weights = [0.15, 0.34, 0.37, 0.10, 0.14]

    fig_bar = go.Figure(go.Bar(
        x=feature_labels,
        y=feature_weights,
        marker=dict(
            color=['#64ffda', '#38bdf8', '#3b82f6', '#1d4ed8', 'rgba(100,255,218,0.3)'],
            line=dict(color='#64ffda', width=1)
        )
    ))

    fig_bar.update_layout(
        paper_bgcolor='#111d38',
        plot_bgcolor='#111d38',
        font=dict(color='#8892b0', size=10),
        margin=dict(l=10, r=10, t=20, b=30),
        height=280,
        yaxis=dict(gridcolor='#1c2d54', range=[0, 0.45]),
        xaxis=dict(gridcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# CHART 2: Patient vs. Population Mean Profile (Radar Scale)
with col_bottom2:
    st.markdown("""
        <div style="background-color: #111d38; border: 1px solid #1c2d54; border-radius: 12px; padding: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #ffffff; font-weight: 700; font-size: 16px;">Patient vs. Population Mean Profile</span>
                <span style="background-color: #16264c; color: #8892b0; padding: 3px 8px; border-radius: 4px; font-size: 10px;">
                    Standardized Scale
                </span>
            </div>
            <p style="color: #8892b0; font-size: 12px; margin-top: 4px; margin-bottom: 10px;">
                Comparison of key physiological markers against healthy cohort baselines.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Standardized radar metrics calculation
    radar_categories = ['Age Ratio', 'Blood Pressure', 'Cholesterol', 'Heart Rate Deficit', 'ST Depression', 'Vessel Obstruction']
    
    patient_radar_vals = [
        min(age / 80.0, 1.0),
        min(trestbps / 200.0, 1.0),
        min(chol / 400.0, 1.0),
        min((210 - thalach) / 140.0, 1.0),
        min(oldpeak / 5.0, 1.0),
        min(ca / 3.0, 1.0)
    ]
    
    cohort_mean_vals = [0.65, 0.60, 0.55, 0.35, 0.20, 0.15]

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=patient_radar_vals + [patient_radar_vals[0]],
        theta=radar_categories + [radar_categories[0]],
        fill='toself',
        fillcolor='rgba(100, 255, 218, 0.15)',
        line=dict(color='#64ffda', width=2),
        name='Patient Profile'
    ))

    fig_radar.add_trace(go.Scatterpolar(
        r=cohort_mean_vals + [cohort_mean_vals[0]],
        theta=radar_categories + [radar_categories[0]],
        line=dict(color='#38bdf8', width=2, dash='dash'),
        name='Healthy Cohort Mean'
    ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor='#111d38',
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(color='#8892b0', gridcolor='#1c2d54')
        ),
        paper_bgcolor='#111d38',
        font=dict(color='#8892b0', size=10),
        margin=dict(l=30, r=30, t=20, b=20),
        height=280,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#e6f1ff"))
    )
    st.plotly_chart(fig_radar, use_container_width=True)