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
        try:
            data = pd.read_excel(uploaded_file)
            if 'DATE' in data.columns:
                data['DATE'] = pd.to_datetime(data['DATE'])
            else:
                st.error("Error: 'DATE' column not found in the uploaded file.")
                return None
            required_columns = [
                'Temperature (F)', 'Dew Point (F)', 'Max Wind Speed (mps)',
                'Avg Wind Speed (mps)', 'Atm Pressure (hPa)', 'Humidity(g/m^3)',
                'Power_Consumption(MU)'
            ]
            if not all(col in data.columns for col in required_columns[:-1]):
                missing_cols = [col for col in required_columns[:-1] if col not in data.columns]
                st.error(f"Error: Missing required columns: {', '.join(missing_cols)} in the uploaded file.")
                return None
            data = data.dropna(subset=required_columns[:-1])
            return data
        except Exception as e:
            st.error(f"An error occurred while loading the data: {e}")
            return None
    return None

@st.cache_data
def predict_missing_power(data):
    """Predicts missing power consumption values."""
    if data is None:
        return pd.DataFrame()  # Return empty DataFrame if no data

    if 'Power_Consumption(MU)' not in data.columns:
        st.error("Error: 'Power_Consumption(MU)' column not found for prediction.")
        return data

    known_data = data[data['Power_Consumption(MU)'].notna()]
    missing_data = data[data['Power_Consumption(MU)'].isna()]
    features = ['Temperature (F)', 'Dew Point (F)', 'Max Wind Speed (mps)',
                'Avg Wind Speed (mps)', 'Atm Pressure (hPa)', 'Humidity(g/m^3)']

    # Check if all features are present in the data
    if not all(feature in known_data.columns for feature in features):
        missing_features = [f for f in features if f not in known_data.columns]
        st.error(f"Error: Missing features for prediction: {', '.join(missing_features)}")
        return data

    X_known = known_data[features]
    y_known = known_data['Power_Consumption(MU)']
    X_missing = missing_data[features]

    if not X_missing.empty and not X_known.empty:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_known, y_known)
        missing_data['Power_Consumption(MU)'] = model.predict(X_missing)
    elif X_missing.empty:
        st.info("No missing power consumption values to predict.")
    elif X_known.empty:
        st.warning("Not enough data with known power consumption to train the prediction model.")

    filled_data = pd.concat([known_data, missing_data]).sort_values(by='DATE') if 'DATE' in data.columns else pd.concat([known_data, missing_data])
    return filled_data

@st.cache_data
def prepare_heatmap_data(data):
    """Prepares data for the heatmap."""

    if data is None or data.empty:
        return pd.DataFrame()

    if 'DATE' not in data.columns:
        st.error("Error: 'DATE' column is required for heatmap generation.")
        return pd.DataFrame()

    data['Month'] = data['DATE'].dt.month
    data['Day'] = data['DATE'].dt.day
    if 'Power_Consumption(MU)' in data.columns:
        data['Power_Consumption(MU)'].fillna(data['Power_Consumption(MU)'].mean(), inplace=True)
        heatmap_data = data.groupby(['Day', 'Month'])['Power_Consumption(MU)'].mean().reset_index()
        heatmap_data_pivot = heatmap_data.pivot(index='Day', columns='Month', values='Power_Consumption(MU)')
        return heatmap_data_pivot
    else:
        st.error("Error: 'Power_Consumption(MU)' column not found for heatmap generation.")
        return pd.DataFrame()

@st.cache_data
def detect_anomalies(data):
    """Detects anomalies in the power consumption data."""
    if data is None or data.empty:
        return pd.DataFrame()

    if 'Power_Consumption(MU)' not in data.columns:
        st.error("Error: 'Power_Consumption(MU)' column is required for anomaly detection.")
        return pd.DataFrame()

    # Handle cases where there might not be enough data for anomaly detection
    if len(data) < 10:  # Example threshold, adjust as needed
        st.warning("Insufficient data for robust anomaly detection.")
        data['anomaly'] = 0  # Mark all as non-anomalous
        return data[data['anomaly'] == -1] # Return empty anomaly DataFrame

    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    data['anomaly'] = iso_forest.fit_predict(data[['Power_Consumption(MU)']])
    anomalies = data[data['anomaly'] == -1]
    return anomalies

def show():
    st.title("Power Consumption Analysis with Anomaly Detection & Heatmap")

    uploaded_file = st.file_uploader("Upload your historical data Excel file with Date, Temperature (F), Dew Point (F), Max Wind Speed (mps),Avg Wind Speed (mps), Atm Pressure (hPa), Humidity(g/m^3), and Power_Consumption(MU)", type=["xlsx"])

    data = load_data(uploaded_file)

    if data is not None:
        filled_data = predict_missing_power(data.copy()) # Use a copy to avoid modifying the original DataFrame
        heatmap_data_pivot = prepare_heatmap_data(filled_data.copy())
        anomalies = detect_anomalies(filled_data.copy())

        # Heatmap
        st.subheader("Heatmap of Predicted/Actual Power Consumption")
        if not heatmap_data_pivot.empty:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.heatmap(heatmap_data_pivot, cmap='YlGnBu', annot=False, fmt=".2f", ax=ax1)
            st.pyplot(fig1)
        else:
            st.warning("No data available to generate heatmap.")

        # Anomaly Detection
        st.subheader("Time Series of Power Consumption with Anomaly Detection")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        if filled_data is not None and 'DATE' in filled_data.columns and 'Power_Consumption(MU)' in filled_data.columns:
          ax2.plot(filled_data['DATE'], filled_data['Power_Consumption(MU)'],
                   label='Power Consumption', color='blue', linewidth=1.5)

          if not anomalies.empty and 'DATE' in anomalies.columns and 'Power_Consumption(MU)' in anomalies.columns:
              ax2.scatter(anomalies['DATE'], anomalies['Power_Consumption(MU)'],
                          color='red', label='Anomaly', s=60, marker='o')

          ax2.set_xlabel('Date')
          ax2.set_ylabel('Power Consumption (MU)')
          ax2.legend()
          ax2.grid(True)
          st.pyplot(fig2)
        else:
            st.warning("Not enough data to plot the time series with anomaly detection.")

        # Display Data and Download
        st.subheader("📁 View Predicted/Actual Power Consumption Data")
        if filled_data is not None:
            st.dataframe(filled_data.head(10))
            st.download_button(
                label="Download Processed Data",
                data=filled_data.to_csv(index=False),
                file_name="Processed_Power_Consumption.csv",
                mime="text/csv"
            )
    else:
        st.info("Please upload an Excel file to start the analysis.")

if _name_ == "main":
    show()
