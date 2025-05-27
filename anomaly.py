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
        # Drop rows with any missing values in the specified columns
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

    # Track originally missing values
    data['was_missing'] = data['Power_Consumption(MU)'].isna()
    known_data = data[data['Power_Consumption(MU)'].notna()]
    missing_data = data[data['Power_Consumption(MU)'].isna()].copy() # Use .copy() to avoid SettingWithCopyWarning

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

    # Filter for values that were originally missing
    missing_data = data[data['was_missing']].copy() # Use .copy()

    if missing_data.empty:
        return pd.DataFrame()

    missing_data['Month'] = missing_data['DATE'].dt.month
    missing_data['Day'] = missing_data['DATE'].dt.day

    heatmap_data = missing_data.groupby(['Day', 'Month'])['Power_Consumption(MU)'].mean().reset_index()
    heatmap_data_pivot = heatmap_data.pivot(index='Day', columns='Month', values='Power_Consumption(MU)')

    return heatmap_data_pivot

@st.cache_data
def detect_anomalies(data):
    """Detects anomalies in the power consumption data."""
    if data is None or data.empty:
        return pd.DataFrame()

    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    # Fit_predict expects a 2D array, so use double brackets for the column
    data['anomaly'] = iso_forest.fit_predict(data[['Power_Consumption(MU)']])
    anomalies = data[data['anomaly'] == -1]
    return anomalies

def show():
    """
    This function encapsulates the entire Streamlit UI and logic
    for the Power Consumption Analysis with Anomaly Detection and
    Missing Data Heatmap. It's designed to be called within a tab
    of a larger Streamlit application.
    """
    st.header("Power Consumption Analysis with Anomaly Detection") # Changed to header for sub-page title

    uploaded_file = st.file_uploader(
        "Upload your historical data Excel file (with 'DATE', 'Power_Consumption(MU)', 'Temperature (F)', 'Dew Point (F)', 'Max Wind Speed (mps)', 'Avg Wind Speed (mps)', 'Atm Pressure (hPa)', 'Humidity(g/m^3)' columns)",
        type=["xlsx"]
    )

    data = load_data(uploaded_file)

    if data is not None:
        filled_data = predict_missing_power(data)
        missing_heatmap_data_pivot = prepare_heatmap_missing_data(filled_data)
        anomalies = detect_anomalies(filled_data)

        st.subheader("📊 Heatmap of Predicted Missing Power Consumption (MU)")
        if not missing_heatmap_data_pivot.empty:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.heatmap(missing_heatmap_data_pivot, cmap='viridis', annot=True, fmt=".2f", ax=ax1, linewidths=.5)
            ax1.set_title("Predicted Missing Power Consumption Heatmap (MU) by Day and Month")
            ax1.set_xlabel("Month")
            ax1.set_ylabel("Day")
            st.pyplot(fig1)
        else:
            st.info("No originally missing power consumption data was found to generate a heatmap.")

        st.subheader("📈 Power Consumption with Anomaly Detection")
        st.write("This chart shows the power consumption over time, with detected anomalies highlighted in red. It also helps visualize how well the predicted missing data points fit the overall pattern.")
        fig2, ax2 = plt.subplots(figsize=(12, 6))

        ax2.plot(filled_data['DATE'], filled_data['Power_Consumption(MU)'],
                 label='Total Power Consumption (Filled)', color='blue', linewidth=1.5, alpha=0.7)

        # Plot originally missing but now filled points separately
        original_missing_points = filled_data[filled_data['was_missing']]
        if not original_missing_points.empty:
            ax2.scatter(original_missing_points['DATE'], original_missing_points['Power_Consumption(MU)'],
                        color='green', label='Originally Missing (Now Predicted)', s=30, alpha=0.8, zorder=5)

        if not anomalies.empty:
            ax2.scatter(anomalies['DATE'], anomalies['Power_Consumption(MU)'],
                        color='red', label='Detected Anomaly', s=70, marker='X', zorder=10)

        ax2.set_xlabel('Date')
        ax2.set_ylabel("Power Consumption (MU)")
        ax2.set_title("Power Consumption Over Time with Predicted Missing Values and Anomalies")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig2)

        st.subheader("📁 Predicted Power Consumption Data (First 10 Rows)")
        st.dataframe(filled_data.head(10))

        st.download_button(
            label="Download All Predicted Data (CSV)",
            data=filled_data.to_csv(index=False).encode('utf-8'),
            file_name="Predicted_Power_Consumption.csv",
            mime="text/csv"
        )
    else:
        st.info("Please upload an Excel file to start the power consumption analysis.")

if __name__ == "__main__"
main()
