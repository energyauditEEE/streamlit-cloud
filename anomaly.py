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
            if 'DATE' not in data.columns:
                st.error("Error: 'DATE' column not found in the uploaded file.")
                return None
            required_columns = [
                'Temperature (F)', 'Dew Point (F)', 'Max Wind Speed (mps)',
                'Avg Wind Speed (mps)', 'Atm Pressure (hPa)', 'Humidity(g/m^3)',
                'Power_Consumption(MU)'
            ]
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return None
            data['DATE'] = pd.to_datetime(data['DATE'])
            data = data.dropna(subset=required_columns[:-1])  # Allow NaNs in Power_Consumption
            return data
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    return None

@st.cache_data
def predict_missing_power(data):
    """Predicts missing power consumption values."""
    if data is None or data.empty:
        return pd.DataFrame()

    if 'Power_Consumption(MU)' not in data.columns:
        st.error("Power consumption column missing")
        return data

    known_data = data[data['Power_Consumption(MU)'].notna()]
    missing_data = data[data['Power_Consumption(MU)'].isna()]
    
    if missing_data.empty:
        return data

    features = ['Temperature (F)', 'Dew Point (F)', 'Max Wind Speed (mps)',
                'Avg Wind Speed (mps)', 'Atm Pressure (hPa)', 'Humidity(g/m^3)']
    
    try:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(known_data[features], known_data['Power_Consumption(MU)'])
        missing_data['Power_Consumption(MU)'] = model.predict(missing_data[features])
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return data

    return pd.concat([known_data, missing_data]).sort_values('DATE')

@st.cache_data
def prepare_heatmap_data(data):
    """Prepares data for the heatmap."""
    if data is None or data.empty:
        return pd.DataFrame()

    try:
        data['Month'] = data['DATE'].dt.month
        data['Day'] = data['DATE'].dt.day
        heatmap_data = data.groupby(['Day', 'Month'])['Power_Consumption(MU)'].mean().reset_index()
        return heatmap_data.pivot(index='Day', columns='Month', values='Power_Consumption(MU)')
    except Exception as e:
        st.error(f"Heatmap generation failed: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def detect_anomalies(data):
    """Detects anomalies in the power consumption data."""
    if data is None or data.empty or 'Power_Consumption(MU)' not in data.columns:
        return pd.DataFrame()

    try:
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        data['anomaly'] = iso_forest.fit_predict(data[['Power_Consumption(MU)']])
        return data[data['anomaly'] == -1]
    except Exception as e:
        st.error(f"Anomaly detection failed: {str(e)}")
        return pd.DataFrame()

def show():
    st.title("Power Consumption Analysis with Anomaly Detection & Heatmap")
    
    uploaded_file = st.file_uploader(
        "Upload your historical data Excel file",
        type=["xlsx"],
        help="Required columns: DATE, Temperature (F), Dew Point (F), Wind Speeds, Pressure, Humidity, Power_Consumption(MU)"
    )

    data = load_data(uploaded_file)
    
    if data is not None:
        filled_data = predict_missing_power(data)
        
        # Heatmap Section
        st.header("Consumption Patterns")
        heatmap_data = prepare_heatmap_data(filled_data)
        if not heatmap_data.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(heatmap_data, cmap='YlGnBu', ax=ax)
            st.pyplot(fig)
        
        # Anomaly Detection Section
        st.header("Anomaly Detection")
        anomalies = detect_anomalies(filled_data)
        if not anomalies.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(filled_data['DATE'], filled_data['Power_Consumption(MU)'], label='Normal')
            ax.scatter(anomalies['DATE'], anomalies['Power_Consumption(MU)'], color='red', label='Anomaly')
            ax.set_xlabel("Date")
            ax.set_ylabel("Power Consumption (MU)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("No anomalies detected")
        
        # Data Preview and Download
        st.header("Processed Data")
        st.dataframe(filled_data.head(10))
        st.download_button(
            "Download Processed Data",
            filled_data.to_csv(index=False),
            "processed_power_data.csv",
            "text/csv"
        )

if __name__ == "__main__":
    main()
