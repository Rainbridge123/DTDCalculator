import streamlit as st
import pandas as pd
from DTD_Calculator import Implied_asset_value

st.set_page_config(page_title="Implied Asset Value Calculator", layout="wide")
st.title("Implied Asset Value & DTD Calculator")

with st.sidebar:
    st.header("Parameters")
    sigma = st.number_input("Sigma", value=4.6940)
    weight = st.number_input("Weight", value=0.3466)
    trading_days = st.number_input("Trading days per year", value=250)
    tol = st.number_input("Tolerance", value=1e-8, format="%.1e")
    max_iter = st.number_input("Max iterations", value=200)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df)

    calc = Implied_asset_value(
        sigma=sigma,
        weight=weight,
        trading_days_per_year=trading_days,
        tol=tol,
        max_iter=max_iter
    )

    result_df = calc.comp_df(df)

    st.subheader("Calculation Result")
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Result CSV",
        csv,
        "implied_asset_value_results.csv",
        "text/csv"
    )
