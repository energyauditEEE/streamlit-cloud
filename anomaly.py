import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest

def show():
    st.header("Anomaly Detection Dashboard")

    # ==============================
    # Part 1: Load Data and Predict Missing Power Consumption
    # ==============================
    try:
        data = pd.read_excel('historical_data.xlsx')  # Replace with your file path
        data['DATE'] = pd.to_datetime(data['DATE'])

        data = data.dropna(subset=[
            'Temperature (F)', 'Dew Point (F)', 'Max Wind Speed (mps)',
            'Avg Wind Speed (mps)', 'Atm Pressure (hPa)', 'Humidity(g/m^3)'
        ])

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

        # ==============================
        # Part 2: Prepare Data for Heatmap
        # ==============================

        missing_data['Month'] = missing_data['DATE'].dt.month
        missing_data['Day'] = missing_data['DATE'].dt.day
        missing_data['Power_Consumption(MU)'].fillna(missing_data['Power_Consumption(MU)'].mean(), inplace=True)

        heatmap_data = missing_data.groupby(['Day', 'Month'])['Power_Consumption(MU)'].mean().reset_index()
        heatmap_data_pivot = heatmap_data.pivot(index='Day', columns='Month', values='Power_Consumption(MU)')

        # ==============================
        # Part 3: Apply Anomaly Detection
        # ==============================

        if not filled_data.empty:
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            filled_data['anomaly'] = iso_forest.fit_predict(filled_data[['Power_Consumption(MU)']])
            anomalies = filled_data[filled_data['anomaly'] == -1]
        else:
            anomalies = pd.DataFrame()

        # ==============================
        # Streamlit Dashboard
        # ==============================

        st.subheader("Heatmap of Predicted Missing Power Consumption")
        if not heatmap_data_pivot.empty:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.heatmap(heatmap_data_pivot, cmap='YlGnBu', annot=True, fmt=".2f", ax=ax1)
            st.pyplot(fig1)
        else:
            st.warning("No data available to generate heatmap.")

        st.subheader("Time Series of Power Consumption with Anomaly Detection")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
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

        st.subheader("📁 View Predicted Power Consumption Data")
        st.dataframe(filled_data.head(10))  # Show first 10 rows
        st.download_button(
            label="Download Predicted Data",
            data=filled_data.to_csv(index=False),
            file_name="Predicted_Power_Consumption.csv",
            mime="text/csv"
        )

        st.info("This interactive dashboard now includes anomaly detection and heatmap visualization!")

    except FileNotFoundError:
        st.error("Error: The file 'historical_data.xlsx' was not found. Please make sure the file is in the correct directory.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    show()