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
                'Power_Consumption(MU)' # Ensure Power_Consumption is also checked for presence
            ]
            # Check for all required columns including Power_Consumption(MU)
            if not all(col in data.columns for col in required_columns):
                missing_cols = [col for col col in required_columns if col not in data.columns]
                st.error(f"Error: Missing required columns: {', '.join(missing_cols)} in the uploaded file.")
                return None

            # Drop rows with any missing values in the features columns before prediction
            data = data.dropna(subset=required_columns[:-1]) # Exclude Power_Consumption(MU) from dropna subset for features
            return data
        except Exception as e:
            st.error(f"An error occurred while loading the data: {e}")
            return None
    return None

@st.cache_data
def predict_missing_power(data):
    """Predicts missing power consumption values."""
    if data is None or data.empty:
        return pd.DataFrame()

    if 'Power_Consumption(MU)' not in data.columns:
        st.error("Error: 'Power_Consumption(MU)' column not found for prediction.")
        return data.copy() # Return a copy to avoid modifying original

    # Track originally missing values before any modification
    data['was_missing'] = data['Power_Consumption(MU)'].isna()

    known_data = data[data['Power_Consumption(MU)'].notna()].copy()
    missing_data = data[data['Power_Consumption(MU)'].isna()].copy()

    features = ['Temperature (F)', 'Dew Point (F)', 'Max Wind Speed (mps)',
                'Avg Wind Speed (mps)', 'Atm Pressure (hPa)', 'Humidity(g/m^3)']

    # Check if all features are present in the known_data (training data)
    if not all(feature in known_data.columns for feature in features):
        missing_features = [f for f in features if f not in known_data.columns]
        st.error(f"Error: Missing features for prediction model training: {', '.join(missing_features)}")
        return data # Return original data if features are missing

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

    # Concatenate known and (potentially) filled missing data
    filled_data = pd.concat([known_data, missing_data]).sort_values(by='DATE') if 'DATE' in data.columns else pd.concat([known_data, missing_data])
    return filled_data

@st.cache_data
def prepare_heatmap_missing_data(data): # Renamed function for clarity
    """Prepares heatmap data for only originally missing dates."""
    if data is None or data.empty:
        return pd.DataFrame()

    if 'was_missing' not in data.columns or 'DATE' not in data.columns or 'Power_Consumption(MU)' not in data.columns:
        st.error("Error: Required columns ('was_missing', 'DATE', 'Power_Consumption(MU)') not found for heatmap generation.")
        return pd.DataFrame()

    # Filter for only the rows that were originally missing
    missing_only_data = data[data['was_missing']].copy()

    if missing_only_data.empty:
        return pd.DataFrame() # Return empty if no missing data was found

    missing_only_data['Month'] = missing_only_data['DATE'].dt.month
    missing_only_data['Day'] = missing_only_data['DATE'].dt.day

    heatmap_data = missing_only_data.groupby(['Day', 'Month'])['Power_Consumption(MU)'].mean().reset_index()
    # Handle potential empty pivot resulting from no data for grouping
    if heatmap_data.empty:
        return pd.DataFrame()

    heatmap_data_pivot = heatmap_data.pivot(index='Day', columns='Month', values='Power_Consumption(MU)')

    return heatmap_data_pivot

@st.cache_data
def detect_anomalies(data):
    """Detects anomalies in the power consumption data."""
    if data is None or data.empty:
        return pd.DataFrame()

    if 'Power_Consumption(MU)' not in data.columns:
        st.error("Error: 'Power_Consumption(MU)' column is required for anomaly detection.")
        return pd.DataFrame()

    # Handle cases where there might not be enough data for anomaly detection
    # IsolationForest needs at least 2 samples, but more for meaningful results.
    if len(data) < 20: # Increased threshold for more robust detection
        st.warning(f"Insufficient data ({len(data)} rows) for robust anomaly detection. At least 20 rows recommended.")
        # If not enough data, assume no anomalies for plotting purposes
        data['anomaly'] = 1
        return pd.DataFrame() # Return empty DataFrame for anomalies

    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    # Ensure the input to fit_predict is a 2D array
    data['anomaly'] = iso_forest.fit_predict(data[['Power_Consumption(MU)']])
    anomalies = data[data['anomaly'] == -1]
    return anomalies

def show():
    st.title("Power Consumption Analysis with Anomaly Detection & Missing Data Heatmap") # Updated title

    uploaded_file = st.file_uploader(
        "Upload your historical data Excel file with 'DATE', 'Power_Consumption(MU)', 'Temperature (F)', 'Dew Point (F)', 'Max Wind Speed (mps)', 'Avg Wind Speed (mps)', 'Atm Pressure (hPa)', 'Humidity(g/m^3)' columns.",
        type=["xlsx"]
    )

    data = load_data(uploaded_file)

    if data is not None:
        # Pass a copy of data to predict_missing_power to ensure 'was_missing' flag
        # is correctly managed if data is processed multiple times
        filled_data = predict_missing_power(data.copy())

        # Only pass data that was originally missing to the heatmap function
        # The 'was_missing' column is crucial here
        missing_heatmap_data_pivot = prepare_heatmap_missing_data(filled_data.copy())
        anomalies = detect_anomalies(filled_data.copy())

        # Heatmap for MISSING Data
        st.subheader("📊 Heatmap of Predicted Power Consumption for Originally Missing Dates (MU)")
        if not missing_heatmap_data_pivot.empty:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.heatmap(missing_heatmap_data_pivot, cmap='YlGnBu', annot=True, fmt=".2f", ax=ax1, linewidths=.5)
            ax1.set_title("Average Predicted Power Consumption on Originally Missing Dates by Day and Month")
            ax1.set_xlabel("Month")
            ax1.set_ylabel("Day")
            st.pyplot(fig1)
        else:
            st.info("No originally missing power consumption data was found, so no specific heatmap for missing data can be generated.")

        # Anomaly Detection Plot
        st.subheader("📈 Time Series of Power Consumption with Anomaly Detection")
        st.write("This chart displays power consumption over time, highlighting detected anomalies and indicating where values were originally missing but have now been predicted.")
        fig2, ax2 = plt.subplots(figsize=(14, 7)) # Larger figure for better visibility

        if filled_data is not None and 'DATE' in filled_data.columns and 'Power_Consumption(MU)' in filled_data.columns:
            ax2.plot(filled_data['DATE'], filled_data['Power_Consumption(MU)'],
                     label='Total Power Consumption (Filled)', color='blue', linewidth=1.5, alpha=0.7)

            # Plot originally missing (now filled) points as green dots
            original_missing_points = filled_data[filled_data['was_missing']]
            if not original_missing_points.empty:
                ax2.scatter(original_missing_points['DATE'], original_missing_points['Power_Consumption(MU)'],
                            color='green', label='Originally Missing (Now Predicted)', s=40, alpha=0.8, zorder=5, marker='o')

            # Plot anomalies as red 'X's
            if not anomalies.empty and 'DATE' in anomalies.columns and 'Power_Consumption(MU)' in anomalies.columns:
                ax2.scatter(anomalies['DATE'], anomalies['Power_Consumption(MU)'],
                            color='red', label='Detected Anomaly', s=100, marker='X', zorder=10)

            ax2.set_xlabel('Date')
            ax2.set_ylabel('Power Consumption (MU)')
            ax2.set_title('Power Consumption Over Time with Predicted Missing Values and Anomalies')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.6)
            plt.xticks(rotation=45) # Rotate x-axis labels for better readability
            plt.tight_layout() # Adjust layout to prevent labels from overlapping
            st.pyplot(fig2)
        else:
            st.warning("Not enough data with required columns to plot the time series with anomaly detection.")

        # Display Data and Download
        st.subheader("📁 View Processed Power Consumption Data")
        if filled_data is not None:
            st.dataframe(filled_data.head(20)) # Show more rows for better initial view
            st.download_button(
                label="Download Processed Data (CSV)",
                data=filled_data.to_csv(index=False).encode('utf-8'),
                file_name="Processed_Power_Consumption.csv",
                mime="text/csv"
            )
    else:
        st.info("Please upload an Excel file to start the analysis.")


if __name__ == "__main__":
    show()
