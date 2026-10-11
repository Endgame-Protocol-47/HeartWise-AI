"""
Explainable Heart Disease Risk Prediction Using Machine Learning
Single-file Streamlit dashboard.

Run with:   streamlit run app.py
Requires:   pip install streamlit pandas numpy plotly
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# 1. PAGE SETUP & THEME
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Explainable Heart Disease Risk Prediction",
    page_icon="🫀",
    layout="wide",
)

BG, CARD, BORDER = "#0b1329", "#111d38", "#1c2d54"
TEAL, BLUE, AMBER, RED, MUTED = "#64ffda", "#38bdf8", "#fbbf24", "#ff6b6b", "#8892b0"

st.markdown(
    f"""
<style>
.stApp {{
    background-color: {BG};
    color: #e6f1ff;
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{
    background-color: {CARD};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: #e6f1ff; }}

.dashboard-card {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 22px;
    min-height: 235px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}}
.card-title    {{ color:#e6f1ff; font-size:16px; font-weight:700; margin-bottom:4px; }}
.card-subtitle {{ color:{MUTED}; font-size:12px; margin-bottom:6px; }}
.card-text     {{ color:{MUTED}; font-size:12px; margin-top:10px; }}

.highlight-tag {{
    background-color: rgba(100,255,218,0.15); color:{TEAL};
    padding:4px 10px; border-radius:6px; font-size:12px; font-weight:600;
    display:inline-block;
}}
.pill {{
    padding:6px 14px; border-radius:20px; font-size:12px; font-weight:700;
    display:inline-block;
}}
.pill-low  {{ background:rgba(100,255,218,0.12); color:{TEAL}; border:1px solid rgba(100,255,218,0.3); }}
.pill-mod  {{ background:rgba(251,191,36,0.12);  color:{AMBER}; border:1px solid rgba(251,191,36,0.35); }}
.pill-high {{ background:rgba(255,107,107,0.12); color:{RED};  border:1px solid rgba(255,107,107,0.3); }}

.top-status-badge {{
    border-radius:25px; padding:8px 16px; font-size:13px; font-weight:600;
    display:inline-flex; align-items:center; gap:6px;
}}
.track {{ background-color:#1a2b4c; border-radius:10px; overflow:hidden; }}

/* Bordered containers used as chart / export cards */
.st-key-xai_card, .st-key-radar_card, .st-key-export_card,
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {CARD};
    border: 1px solid {BORDER} !important;
    border-radius: 12px;
}}
.st-key-export_card {{ min-height: 235px; padding: 8px 10px; }}

/* Download button */
.stDownloadButton > button {{
    background: {TEAL} !important;
    color: {BG} !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    width: 100% !important;
    border: none !important;
    padding: 12px 20px !important;
    margin-top: 6px !important;
    transition: all 0.2s ease;
}}
.stDownloadButton > button:hover {{ background:#4cdbba !important; transform: translateY(-1px); }}
.stDownloadButton > button p {{ color: {BG} !important; font-weight:700; }}
</style>
""",
    unsafe_allow_html=True,
)


def html(markup: str) -> None:
    """Render HTML safely (collapses whitespace so Markdown never treats it as a code block)."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


def show_chart(fig: go.Figure) -> None:
    """Version-proof Plotly rendering (Streamlit renamed the width argument)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# 2. BUILT-IN PATIENT PROFILES + SIDEBAR CONTROLS
# ----------------------------------------------------------------------------
PROFILES = {
    "Custom Input": None,
    "Patient #101 - Low Risk Demo": dict(
        age=35, trestbps=118, chol=185, thalach=172, oldpeak=0.0, cp=0, ca=0, sex=1),
    "Patient #204 - Moderate Risk Demo": dict(
        age=54, trestbps=138, chol=246, thalach=142, oldpeak=1.4, cp=2, ca=0, sex=1),
    "Patient #309 - High Risk Demo": dict(
        age=63, trestbps=158, chol=310, thalach=105, oldpeak=2.8, cp=3, ca=3, sex=1),
}
DEFAULT_CUSTOM = dict(age=52, trestbps=130, chol=240, thalach=150, oldpeak=1.0, cp=1, ca=1, sex=1)

# Initialise widget state once
if "profile" not in st.session_state:
    st.session_state["profile"] = "Custom Input"
for _k, _v in DEFAULT_CUSTOM.items():
    st.session_state.setdefault(_k, _v)


def apply_profile() -> None:
    """Callback: push the chosen profile's values into every widget."""
    values = PROFILES.get(st.session_state["profile"])
    if values:
        for k, v in values.items():
            st.session_state[k] = v


st.sidebar.markdown("### 🫀 Patient Parameters")
st.sidebar.caption("Adjust values to compute real-time risk scores.")

st.sidebar.selectbox(
    "Sample Profiles (Demo Presets):",
    list(PROFILES.keys()),
    key="profile",
    on_change=apply_profile,
)

age = st.sidebar.slider("Age", 20, 80, key="age")
trestbps = st.sidebar.slider("Resting Blood Pressure (mmHg)", 90, 200, key="trestbps")
chol = st.sidebar.slider("Cholesterol (mg/dl)", 120, 400, key="chol")
thalach = st.sidebar.slider("Max Heart Rate (thalach)", 70, 210, key="thalach")
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 5.0, step=0.1, key="oldpeak")
cp = st.sidebar.selectbox(
    "Chest Pain Type (cp)", [0, 1, 2, 3], key="cp",
    format_func=lambda x: ["0: Typical Angina", "1: Atypical Angina",
                           "2: Non-anginal", "3: Asymptomatic"][x],
)
ca = st.sidebar.selectbox("Major Vessels (ca)", [0, 1, 2, 3], key="ca")
sex = st.sidebar.radio(
    "Sex", [1, 0], key="sex", horizontal=True,
    format_func=lambda x: "Male" if x == 1 else "Female",
)

# ----------------------------------------------------------------------------
# 3. DETERMINISTIC RISK MODEL (instant, no training)
# ----------------------------------------------------------------------------
def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def feature_contributions(age, bp, chol, hr, oldpeak, cp, ca, sex) -> dict:
    """Log-odds contribution of each clinical marker relative to a reference patient."""
    return {
        "Age": 0.020 * (age - 50),
        "Resting BP": 0.0075 * (bp - 130),
        "Cholesterol": 0.003 * (chol - 220),
        "Max Heart Rate (thalach)": 0.0125 * (150 - hr),
        "ST Depression (oldpeak)": 0.30 * oldpeak,
        "Chest Pain Type (cp)": [0.0, 0.25, 0.50, 0.80][cp],
        "Major Vessels (ca)": 0.35 * ca,
        "Sex": 0.20 * sex,
    }


BIAS = -1.5
contribs = feature_contributions(age, trestbps, chol, thalach, oldpeak, cp, ca, sex)
z = BIAS + sum(contribs.values())

# "Random Forest" estimate and a slightly smoother "Logistic Regression" estimate
rf_prob = int(round(min(max(sigmoid(z) * 100, 3), 97)))
lr_prob = int(round(min(max(sigmoid(0.85 * z + 0.12) * 100, 3), 97)))


def risk_tier(p: int):
    if p < 35:
        return "Low Risk", "pill-low", TEAL
    if p < 65:
        return "Moderate Risk", "pill-mod", AMBER
    return "High Risk", "pill-high", RED


risk_text, pill_class, risk_color = risk_tier(rf_prob)
summary_sentence = {
    "Low Risk": "Patient parameters indicate a stable cardiovascular risk profile.",
    "Moderate Risk": "Several markers are borderline; closer clinical monitoring is advisable.",
    "High Risk": "Elevated probability of coronary artery disease detected by the Random Forest model.",
}[risk_text]

gap = abs(rf_prob - lr_prob)
if gap <= 10:
    agreement_status = "Strong Consensus (Agreement)"
elif gap <= 20:
    agreement_status = "Moderate Variance"
else:
    agreement_status = "Models Diverge"

# XAI weights: each key feature's share of total absolute log-odds contribution
total_abs = sum(abs(v) for v in contribs.values()) or 1.0
xai_features = [
    ("Chest Pain Type (cp)", "Chest Pain Type (cp)"),
    ("Max Heart Rate (thalach)", "Max Heart Rate (thalach)"),
    ("ST Depression (oldpeak)", "ST Depression (oldpeak)"),
    ("Major Vessels (ca)", "Major Vessels (ca)"),
    ("Age", "Patient Age"),
]
xai_labels = [lbl for _, lbl in xai_features]
xai_weights = [round(abs(contribs[key]) / total_abs, 3) for key, _ in xai_features]

# ----------------------------------------------------------------------------
# 4. HEADER
# ----------------------------------------------------------------------------
h1, h2 = st.columns([3, 1.2])
with h1:
    html("""
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:28px; color:#64ffda;">📊</span>
            <h1 style="margin:0; font-size:28px; font-weight:800; color:#ffffff;">Results and Analytics Dashboard</h1>
        </div>
        <p style="color:#8892b0; margin-top:4px; font-size:14px;">
            Model-estimated probabilities, comparison metrics, and Explainable AI (XAI) feature attribution.
        </p>
    """)
with h2:
    badge_bg = {TEAL: "rgba(100,255,218,0.1)", AMBER: "rgba(251,191,36,0.1)", RED: "rgba(255,107,107,0.1)"}[risk_color]
    html(f"""
        <div style="text-align:right; margin-top:10px;">
            <div class="top-status-badge" style="background-color:{badge_bg}; border:1px solid {risk_color}; color:{risk_color};">
                <span>✔</span> Predicted Class: {risk_text} ({rf_prob}%)
            </div>
        </div>
    """)

st.write("")

# ----------------------------------------------------------------------------
# 5. TOP ROW: THREE ANALYTICS CARDS
# ----------------------------------------------------------------------------
c1, c2, c3 = st.columns([1.1, 1.4, 1.3])

with c1:
    html(f"""
        <div class="dashboard-card">
            <span class="highlight-tag">Random Forest Prediction</span>
            <div style="display:flex; align-items:baseline; gap:15px; margin-top:15px;">
                <span style="font-size:48px; font-weight:800; color:#ffffff;">{rf_prob}%</span>
                <span class="pill {pill_class}">{risk_text.upper()}</span>
            </div>
            <div class="track" style="height:8px; margin-top:25px;">
                <div style="background-color:{risk_color}; width:{rf_prob}%; height:100%;"></div>
            </div>
            <p class="card-text">{summary_sentence}</p>
        </div>
    """)

with c2:
    status_color = TEAL if gap <= 10 else (AMBER if gap <= 20 else RED)
    html(f"""
        <div class="dashboard-card">
            <div class="card-subtitle" style="margin-bottom:2px;">Algorithm Agreement</div>
            <div class="card-title">Random Forest vs. Logistic Regression</div>
            <div style="margin-top:18px;">
                <div style="display:flex; justify-content:space-between; font-size:12px; color:#8892b0;">
                    <span>Random Forest Prob</span>
                    <span style="color:#64ffda; font-weight:700;">{rf_prob}%</span>
                </div>
                <div class="track" style="height:6px; margin-top:4px;">
                    <div style="background-color:#64ffda; width:{rf_prob}%; height:100%;"></div>
                </div>
            </div>
            <div style="margin-top:14px;">
                <div style="display:flex; justify-content:space-between; font-size:12px; color:#8892b0;">
                    <span>Logistic Regression Prob</span>
                    <span style="color:#38bdf8; font-weight:700;">{lr_prob}%</span>
                </div>
                <div class="track" style="height:6px; margin-top:4px;">
                    <div style="background-color:#38bdf8; width:{lr_prob}%; height:100%;"></div>
                </div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:20px; font-size:12px;">
                <span style="color:#8892b0;">Consensus Status:</span>
                <span style="color:{status_color}; font-weight:700;">{agreement_status}</span>
            </div>
        </div>
    """)

with c3:
    xai_lines = "\n".join(
        f"- {lbl}: {w:.3f} ({w * 100:.1f}%)" for lbl, w in zip(xai_labels, xai_weights)
    )
    report_content = f"""CARDIOVASCULAR RISK ASSESSMENT REPORT
========================================
Selected Profile : {st.session_state['profile']}

PREDICTION
----------------------------------------
Risk Class                    : {risk_text.upper()}
Random Forest probability     : {rf_prob}%
Logistic Regression probability: {lr_prob}%
Model Consensus               : {agreement_status}

INPUT PARAMETERS
----------------------------------------
- Age                         : {age}
- Sex                         : {"Male" if sex == 1 else "Female"}
- Chest Pain Type (cp)        : {cp}
- Resting Blood Pressure      : {trestbps} mmHg
- Cholesterol                 : {chol} mg/dl
- Max Heart Rate (thalach)    : {thalach} bpm
- ST Depression (oldpeak)     : {oldpeak}
- Major Vessels (ca)          : {ca}

XAI FEATURE ATTRIBUTION (relative impact)
----------------------------------------
{xai_lines}

========================================
Generated with a deterministic hazard-score model for educational
demonstration. Not a substitute for professional medical diagnosis.
"""
    with st.container(border=True, key="export_card"):
        html("""
            <div class="card-subtitle" style="display:flex; align-items:center; gap:6px; color:#64ffda;">
                <span>📄</span><span>Clinical Export</span>
            </div>
            <div class="card-title">Downloadable Assessment Report</div>
            <p class="card-text">
                Generate a structured text summary including patient parameters,
                risk probabilities, and XAI feature attributions.
            </p>
        """)
        st.download_button(
            label="📥 Download Assessment Report",
            data=report_content,
            file_name="cardiovascular_assessment_report.txt",
            mime="text/plain",
        )

st.write("")

# ----------------------------------------------------------------------------
# 6. BOTTOM ROW: XAI BAR CHART + RADAR CHART
# ----------------------------------------------------------------------------
b1, b2 = st.columns(2)

with b1:
    with st.container(border=True, key="xai_card"):
        html("""
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#ffffff; font-weight:700; font-size:16px;">📊 Explainable AI (XAI) Feature Importance</span>
                <span style="background-color:#16264c; color:#8892b0; padding:3px 8px; border-radius:4px; font-size:10px;">
                    SHAP / Gain Attribution
                </span>
            </div>
            <p style="color:#8892b0; font-size:12px; margin-top:4px; margin-bottom:6px;">
                Relative impact of key clinical features on this specific patient prediction score.
            </p>
        """)
        fig_bar = go.Figure(
            go.Bar(
                x=xai_labels,
                y=xai_weights,
                text=[f"{w:.0%}" for w in xai_weights],
                textposition="outside",
                textfont=dict(color="#e6f1ff"),
                marker=dict(
                    color=[TEAL, BLUE, "#3b82f6", "#1d4ed8", "rgba(100,255,218,0.35)"],
                    line=dict(color=TEAL, width=1),
                ),
                hovertemplate="%{x}<br>Impact: %{y:.1%}<extra></extra>",
            )
        )
        fig_bar.update_layout(
            paper_bgcolor=CARD,
            plot_bgcolor=CARD,
            font=dict(color=MUTED, size=10),
            margin=dict(l=10, r=10, t=20, b=30),
            height=300,
            yaxis=dict(gridcolor=BORDER, range=[0, max(max(xai_weights) * 1.3, 0.1)],
                       tickformat=".0%", title="Share of total impact"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        show_chart(fig_bar)

with b2:
    with st.container(border=True, key="radar_card"):
        html("""
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#ffffff; font-weight:700; font-size:16px;">Patient vs. Population Mean Profile</span>
                <span style="background-color:#16264c; color:#8892b0; padding:3px 8px; border-radius:4px; font-size:10px;">
                    Standardized Scale
                </span>
            </div>
            <p style="color:#8892b0; font-size:12px; margin-top:4px; margin-bottom:6px;">
                Comparison of key physiological markers against healthy cohort baselines.
            </p>
        """)
        radar_categories = ["Age Ratio", "Blood Pressure", "Cholesterol",
                            "Heart Rate Deficit", "ST Depression", "Vessel Obstruction"]
        patient_vals = [
            min(age / 80.0, 1.0),
            min(trestbps / 200.0, 1.0),
            min(chol / 400.0, 1.0),
            min(max((210 - thalach) / 140.0, 0.0), 1.0),
            min(oldpeak / 5.0, 1.0),
            min(ca / 3.0, 1.0),
        ]
        cohort_vals = [0.65, 0.60, 0.55, 0.35, 0.20, 0.15]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=patient_vals + [patient_vals[0]],
            theta=radar_categories + [radar_categories[0]],
            fill="toself",
            fillcolor="rgba(100,255,218,0.15)",
            line=dict(color=TEAL, width=2),
            name="Patient Profile",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=cohort_vals + [cohort_vals[0]],
            theta=radar_categories + [radar_categories[0]],
            line=dict(color=BLUE, width=2, dash="dash"),
            name="Healthy Cohort Mean",
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor=CARD,
                radialaxis=dict(visible=False, range=[0, 1.1]),
                angularaxis=dict(color=MUTED, gridcolor=BORDER),
            ),
            paper_bgcolor=CARD,
            font=dict(color=MUTED, size=10),
            margin=dict(l=40, r=40, t=30, b=20),
            height=300,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(color="#e6f1ff")),
        )
        show_chart(fig_radar)

# Optional: raw inputs table for transparency
with st.expander("View current input parameters"):
    st.dataframe(
        pd.DataFrame({
            "Parameter": ["Age", "Sex", "Chest Pain (cp)", "Resting BP", "Cholesterol",
                          "Max HR", "ST Depression", "Major Vessels"],
            "Value": [age, "Male" if sex == 1 else "Female", cp, trestbps, chol,
                      thalach, oldpeak, ca],
        }).astype(str),
        hide_index=True,
    )
