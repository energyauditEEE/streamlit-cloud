import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest


@st.cache_data
def load_data(uploaded_file):
    """Loads and preprocesses the data."""
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        data['DATE'] = pd.to_datetime(data['DATE'])
        data = data.dropna(subset=[
            'Temperature (F)', 'Dew Point (F)', 'Max Wind Speed (mps)',
            'Avg Wind Speed (mps)', 'Atm Pressure (hPa)', 'Humidity(g/m^3)'
        ])
        return data
    return None


@st.cache_data
def predict_missing_power(data):
    """Predicts missing power consumption values."""
    if data is None:
        return pd.DataFrame()

    data['was_missing'] = data['Power_Consumption(MU)'].isna()  # Track originally missing values
    known_data = data[data['Power_Consumption(MU)'].notna()]
    missing_data = data[data['Power_Consumption(MU)'].isna()]
    features = ['Temperature (F)', 'Dew Point (F)', 'Max Wind Speed (mps)',
                'Avg Wind Speed (mps)', 'Atm Pressure (hPa)', 'Humidity(g/m^3)']

    X_known = known_data[features]
    y_known = known_data['Power_Consumption(MU)']
    X_missing = missing_data[features]

    if not X_missing.empty:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_known, y_known)
        missing_data['Power_Consumption(MU)'] = model.predict(X_missing)

    filled_data = pd.concat([known_data, missing_data]).sort_values(by='DATE')
    return filled_data


@st.cache_data
def prepare_heatmap_missing_data(data):
    """Prepares heatmap data for only originally missing dates."""
    if data is None or data.empty:
        return pd.DataFrame()

    missing_data = data[data['was_missing']]  # Use the flag instead of checking NaN
    missing_data['Month'] = missing_data['DATE'].dt.month
    missing_data['Day'] = missing_data['DATE'].dt.day

    if missing_data.empty:
        return pd.DataFrame()

    heatmap_data = missing_data.groupby(['Day', 'Month'])['Power_Consumption(MU)'].mean().reset_index()
    heatmap_data_pivot = heatmap_data.pivot(index='Day', columns='Month', values='Power_Consumption(MU)')

    return heatmap_data_pivot


@st.cache_data
def detect_anomalies(data):
    """Detects anomalies in the power consumption data."""
    if data is None or data.empty:
        return pd.DataFrame()

    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    data['anomaly'] = iso_forest.fit_predict(data[['Power_Consumption(MU)']])
    anomalies = data[data['anomaly'] == -1]
    return anomalies


# Streamlit App
st.title("Power Consumption Analysis with Anomaly Detection & Missing Data Heatmap")

uploaded_file = st.file_uploader(
    "Upload your historical data Excel file with Date, Temperature (F), Dew Point (F), Max Wind Speed (mps), Avg Wind Speed (mps), Atm Pressure (hPa), Humidity(g/m^3)",
    type=["xlsx"])

data = load_data(uploaded_file)

if data is not None:
    filled_data = predict_missing_power(data)
    missing_heatmap_data_pivot = prepare_heatmap_missing_data(filled_data)
    anomalies = detect_anomalies(filled_data)

    # Heatmap for Missing Data
    st.subheader("Heatmap of Missing Power Consumption Dates (MU)")
    if not missing_heatmap_data_pivot.empty:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.heatmap(missing_heatmap_data_pivot, cmap='YlGnBu', annot=True, fmt=".2f", ax=ax1)
        ax1.set_title("Missing Power Consumption Heatmap (MU)")
        st.pyplot(fig1)
    else:
        st.warning("No missing data available to generate heatmap.")

    # Anomaly Detection
    st.subheader(" Power Consumption with Anomaly Detection to check the fit-in of missing datas")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    if filled_data is not None:
        ax2.plot(filled_data['DATE'], filled_data['Power_Consumption(MU)'],
                 label='Power Consumption', color='blue', linewidth=1.5)

        if not anomalies.empty:
            ax2.scatter(anomalies['DATE'], anomalies['Power_Consumption(MU)'],
                        color='red', label='Anomaly', s=60, marker='o')

        ax2.set_xlabel('Date')
        ax2.set_ylabel("Power Consumption (MU)")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)


        # Display Data and Download
        st.subheader("📁 View Predicted Power Consumption Data")
        st.dataframe(filled_data.head(10))
        st.download_button(
            label="Download Predicted Data",
            data=filled_data.to_csv(index=False),
            file_name="Predicted_Power_Consumption.csv",
            mime="text/csv"
        )
    else:  # Ensure it's aligned correctly
        st.info("Please upload an Excel file to start the analysis.")
