
# ============================================================
# EXPLAINABLE HEART DISEASE RISK ASSESSMENT
# Python + Streamlit + Machine Learning
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# -------------------- PAGE SETTINGS -------------------------

st.set_page_config(
    page_title="CardioInsight | Heart Risk AI",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426, #102c49);
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background-color: #091b30;
}
h1, h2, h3 {
    color: #64d8ff !important;
}
div[data-testid="stMetric"] {
    background: #122d49;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #244967;
}
.stButton > button {
    background-color: #00a6c7;
    color: white;
    border-radius: 9px;
    border: none;
    font-weight: bold;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar > 120 mg/dL",
    "restecg": "Resting ECG result",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression",
    "slope": "Exercise ST slope",
    "ca": "Major vessels",
    "thal": "Thalassemia test category"
}

# -------------------- LOAD DATASET ---------------------------

@st.cache_data
def load_data():
    url = (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/heart-disease/"
        "processed.cleveland.data"
    )

    columns = FEATURES + ["target"]

    df = pd.read_csv(
        url,
        header=None,
        names=columns,
        na_values="?"
    )

    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=columns).copy()

    # Original target: 0 = no disease; 1-4 = disease
    df["target"] = (df["target"] > 0).astype(int)

    if len(df) < 30 or df["target"].nunique() != 2:
        raise ValueError("Dataset does not contain enough valid records.")

    return df


try:
    data = load_data()
except Exception as error:
    st.error("The heart disease dataset could not be loaded.")
    st.write("Details:", str(error))
    st.info(
        "Check your internet connection. The application downloads "
        "the public UCI dataset when it starts."
    )
    st.stop()

# -------------------- TRAIN MODELS ---------------------------

@st.cache_resource
def train_models(dataset):
    X = dataset[FEATURES]
    y = dataset["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ])

    forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    logistic.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    models = {
        "Logistic Regression": logistic,
        "Random Forest": forest
    }

    results = []

    for name, model in models.items():
        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            )
        })

    return models, pd.DataFrame(results), X_test

models, metrics, X_test = train_models(data)

# -------------------- SIDEBAR -------------------------------

st.sidebar.title("❤️ CardioInsight")
st.sidebar.caption("Explainable Heart Risk Assessment")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Patient Assessment", "Model Analytics", "About"]
)

st.sidebar.divider()
st.sidebar.caption("Academic demonstration only.")

# -------------------- DASHBOARD -----------------------------

if page == "Dashboard":
    st.title("CardioInsight AI")
    st.subheader("Explainable Heart Disease Risk Prediction")
    st.write(
        "Explore a machine-learning demonstration that analyses "
        "clinical features and compares two classification models."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Records", len(data))
    c2.metric("Clinical Features", len(FEATURES))
    c3.metric("Models Trained", len(models))

    st.divider()
    st.subheader("Model Performance")

    display_metrics = metrics.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        display_metrics[col] = (
            display_metrics[col] * 100
        ).round(1).astype(str) + "%"

    st.dataframe(display_metrics, use_container_width=True)

    chart = px.bar(
        metrics.melt(
            id_vars="Model",
            var_name="Metric",
            value_name="Score"
        ),
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Comparison of Machine Learning Models",
        range_y=[0, 1],
        template="plotly_dark"
    )
    st.plotly_chart(chart, use_container_width=True)

    st.info(
        "Select Patient Assessment in the sidebar to try the interactive demo."
    )

# -------------------- PATIENT ASSESSMENT --------------------

elif page == "Patient Assessment":
    st.title("Patient Risk Assessment")
    st.write("Enter clinical values or load a sample profile.")

    sample = st.selectbox(
        "Choose a sample profile",
        ["Custom Input", "Sample A", "Sample B"]
    )

    if sample == "Sample A":
        defaults = [55, 1, 2, 130, 240, 0, 1, 150, 0, 1.0, 2, 0, 3]
    elif sample == "Sample B":
        defaults = [63, 1, 4, 145, 280, 0, 2, 120, 1, 2.5, 2, 2, 7]
    else:
        defaults = [45, 0, 2, 120, 210, 0, 0, 165, 0, 0.0, 2, 0, 3]

    with st.form("patient_form"):
        st.subheader("Patient Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age (years)", 18, 100, defaults[0]
            )
            sex = st.selectbox(
                "Sex (dataset coding)",
                [0, 1],
                index=defaults[1],
                format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
            )
            cp = st.selectbox(
                "Chest pain type (1–4)",
                [1, 2, 3, 4],
                index=[1, 2, 3, 4].index(defaults[2])
            )
            bp = st.number_input(
                "Resting blood pressure (mmHg)",
                70, 250, defaults[3]
            )
            chol = st.number_input(
                "Cholesterol (mg/dL)",
                80, 700, defaults[4]
            )

        with col2:
            fbs = st.selectbox(
                "Fasting blood sugar > 120 mg/dL?",
                [0, 1],
                index=defaults[5],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )
            restecg = st.selectbox(
                "Resting ECG category",
                [0, 1, 2],
                index=defaults[6]
            )
            thalach = st.number_input(
                "Maximum heart rate achieved",
                50, 250, defaults[7]
            )
            exang = st.selectbox(
                "Exercise-induced angina?",
                [0, 1],
                index=defaults[8],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

        with col3:
            oldpeak = st.number_input(
                "ST depression (oldpeak)",
                0.0, 10.0, float(defaults[9]), step=0.1
            )
            slope = st.selectbox(
                "Exercise ST slope (1–3)",
                [1, 2, 3],
                index=[1, 2, 3].index(defaults[10])
            )
            ca = st.selectbox(
                "Major vessels (0–3)",
                [0, 1, 2, 3],
                index=defaults[11]
            )
            thal = st.selectbox(
                "Thal test category",
                [3, 6, 7],
                index=[3, 6, 7].index(defaults[12])
            )

        submitted = st.form_submit_button("Assess Patient")

    if submitted:
        patient = pd.DataFrame([[
            age, sex, cp, bp, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]], columns=FEATURES)

        st.divider()
        st.subheader("Model Results")

        report = {
            "Assessment": "Educational model output",
            "Age": age,
            "Blood Pressure": bp,
            "Cholesterol": chol
        }

        for name, model in models.items():
            prediction = int(model.predict(patient)[0])
            probability = float(model.predict_proba(patient)[0][1])

            st.markdown(f"### {name}")
            a, b = st.columns(2)
            a.metric("Predicted Class", (
                "Positive" if prediction == 1 else "Negative"
            ))
            b.metric("Model-estimated probability", f"{probability:.1%}")

            report[name + " predicted class"] = prediction
            report[name + " estimated probability"] = round(
                probability, 4
            )

        st.warning(
            "These are educational machine-learning outputs, not a "
            "medical diagnosis or a validated estimate of your personal "
            "future risk. Consult a qualified healthcare professional."
        )

        st.download_button(
            "Download Assessment Report",
            data=pd.Series(report).to_csv(header=False),
            file_name="heart_assessment_report.csv",
            mime="text/csv"
        )

# -------------------- MODEL ANALYTICS -----------------------

elif page == "Model Analytics":
    st.title("Model Analytics")
    st.write("Evaluate the models using a held-out test dataset.")

    st.subheader("Performance Metrics")
    st.dataframe(
        metrics.style.format({
            "Accuracy": "{:.1%}",
            "Precision": "{:.1%}",
            "Recall": "{:.1%}",
            "F1 Score": "{:.1%}"
        }),
        use_container_width=True
    )

    st.subheader("Random Forest Feature Importance")
    forest = models["Random Forest"].named_steps["model"]

    importance = pd.DataFrame({
        "Feature": [FEATURE_LABELS[x] for x in FEATURES],
        "Importance": forest.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Features Used by Random Forest",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Feature importance describes the model's use of features. "
        "It does not prove that a feature causes heart disease."
    )

    st.download_button(
        "Download Model Metrics",
        data=metrics.to_csv(index=False),
        file_name="model_metrics.csv",
        mime="text/csv"
    )

# -------------------- ABOUT ---------------------------------

elif page == "About":
    st.title("About This Project")

    st.markdown("""
    ### Project Objective
    Demonstrate heart disease classification using machine learning.

    ### Algorithms
    - **Logistic Regression:** a linear classification model.
    - **Random Forest:** a model that combines multiple decision trees.

    ### Workflow
    1. Load and clean a public heart disease dataset.
    2. Split the data into training and testing sets.
    3. Train both machine learning models.
    4. Compare accuracy, precision, recall and F1 score.
    5. Explore feature importance and test sample patient inputs.

    ### Dataset
    UCI Cleveland Heart Disease dataset.

    ### Important Limitation
    This is a student demonstration, not a clinical decision-support
    system. Its predictions must not be used to diagnose, treat,
    or rule out heart disease.
    """)

st.sidebar.divider()
st.sidebar.caption("CardioInsight AI | Student Project")

