# anomaly.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest

@st.cache_data
def load_data(uploaded_file):
    """Loads and preprocesses data with enhanced error handling"""
    if uploaded_file is None:
        return None
        
    try:
        data = pd.read_excel(uploaded_file)
        required_cols = [
            'DATE', 'Temperature (F)', 'Dew Point (F)',
            'Max Wind Speed (mps)', 'Avg Wind Speed (mps)',
            'Atm Pressure (hPa)', 'Humidity(g/m^3)', 'Power_Consumption(MU)'
        ]
        
        # Validate columns
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None
            
        # Convert and clean data
        data['DATE'] = pd.to_datetime(data['DATE'], errors='coerce')
        data = data.dropna(subset=required_cols[1:-1])  # Allow NaNs in Power_Consumption
        
        return data
        
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None

@st.cache_data
def prepare_heatmap_data(data):
    """Prepares data for the heatmap."""

    if data is None or data.empty:
        return pd.DataFrame()

    data['Month'] = data['DATE'].dt.month
    data['Day'] = data['DATE'].dt.day
    data['Power_Consumption(MU)'].fillna(data['Power_Consumption(MU)'].mean(), inplace=True)
    heatmap_data = data.groupby(['Day', 'Month'])['Power_Consumption(MU)'].mean().reset_index()
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
st.title("Power Consumption Analysis with Anomaly Detection & Heatmap")

uploaded_file = st.file_uploader("Upload your historical data Excel file with Date, Temperature (F), Dew Point (F), Max Wind Speed (mps),Avg Wind Speed (mps), Atm Pressure (hPa), Humidity(g/m^3) ", type=["xlsx"])

data = load_data(uploaded_file)

if data is not None:
    filled_data = predict_missing_power(data)
    heatmap_data_pivot = prepare_heatmap_data(filled_data)
    anomalies = detect_anomalies(filled_data)

    # Heatmap
    st.subheader("Heatmap of Predicted Missing Power Consumption")
    if not heatmap_data_pivot.empty:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data_pivot, cmap='YlGnBu', annot=False, fmt=".2f", ax=ax1)
        st.pyplot(fig1)
    else:
        st.warning("No data available to generate heatmap.")

    # Anomaly Detection
    st.subheader("Time Series of Power Consumption with Anomaly Detection")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    if filled_data is not None:
      ax2.plot(filled_data['DATE'], filled_data['Power_Consumption(MU)'],
               label='Power Consumption', color='blue', linewidth=1.5)

      if not anomalies.empty:
          ax2.scatter(anomalies['DATE'], anomalies['Power_Consumption(MU)'],
                      color='red', label='Anomaly', s=60, marker='o')

      ax2.set_xlabel('Date')
      ax2.set_ylabel('Power Consumption (MU)')
      ax2.legend()
      ax2.grid(True)
      st.pyplot(fig2)

    # Display Data and Download
    st.subheader("📁 View Predicted Power Consumption Data")
    if filled_data is not None:
        st.dataframe(filled_data.head(10))
        st.download_button(
            label="Download Predicted Data",
            data=filled_data.to_csv(index=False),
            file_name="Predicted_Power_Consumption.csv",
            mime="text/csv"
        )

# For local testing (remove in deployment)
if __name__ == "__main__":
    show()
