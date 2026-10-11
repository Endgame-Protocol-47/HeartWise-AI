
import streamlit as st
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="HeartWise AI",
    page_icon="❤️",
    layout="wide"
)

# Load the dataset
st.title("❤️ HeartWise AI")
st.subheader("Explainable Heart Disease Risk Prediction")

st.write(
    "Welcome to HeartWise AI. "
    "This project explores clinical data for heart disease research."
)

data_path = Path(__file__).parent / "heart.csv"

if data_path.exists():
    df = pd.read_csv(data_path)

    st.success("Dataset loaded successfully!")

    # Dataset overview
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Column Information")
    st.write("Columns in your dataset:")
    st.write(df.columns.tolist())

    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)

else:
    st.error(
        "heart.csv was not found. "
        "Place heart.csv in the same folder as app.py."
    )
