import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show():
    st.header("Wind Energy Dashboard")

    # Streamlit Title
    st.subheader("Wind Energy Prediction")

    # File Upload Section
    uploaded_file = st.file_uploader("Upload your wind speed data Excel file with DATE, max wind speed(mps)", type=["xlsx"])

    if uploaded_file:
        try:
            # Load the dataset
            data = pd.read_excel(uploaded_file, sheet_name='Sheet1')
            if 'DATE' not in data.columns or 'Max Wind Speed (mps)' not in data.columns:
                st.error("Error: The uploaded file must contain 'DATE' and 'Max Wind Speed (mps)' columns.")
                return
            data['DATE'] = pd.to_datetime(data['DATE'])

            # User Inputs for Turbine Parameters
            num_turbines = st.number_input("Enter the number of turbines:", min_value=1, value=10)
            swept_area = st.number_input("Enter the swept area of each turbine (in m²):", min_value=1.0, value=50.0)
            efficiency = st.slider("Enter turbine efficiency (as a percentage):", min_value=1, max_value=100, value=40) / 100
            air_density = 1.225  # kg/m³ (constant air density)

            predict_button = st.button("Predict Wind Energy")

            if predict_button:
                # Predicting Wind Energy for 2025-2030
                future_dates = pd.date_range(start='2025-01-01', end='2030-12-31', freq='D')
                # Ensure 'Max Wind Speed (mps)' is in the uploaded data before using it
                if not data['Max Wind Speed (mps)'].empty:
                    future_data = pd.DataFrame({
                        'DATE': future_dates,
                        'Max Wind Speed (mps)': np.random.choice(data['Max Wind Speed (mps)'].fillna(data['Max Wind Speed (mps)'].mean()), size=len(future_dates))
                    })

                    # Wind Energy Calculation
                    future_data['Wind Power (W)'] = 0.5 * air_density * swept_area * (future_data['Max Wind Speed (mps)'] ** 3) * efficiency
                    future_data['Energy per Turbine (kWh/day)'] = future_data['Wind Power (W)'] * 24 / 1000  # Convert to kWh/day
                    future_data['Total Energy (kWh/day)'] = future_data['Energy per Turbine (kWh/day)'] * num_turbines

                    # Display Predictions
                    st.subheader("Predicted Total Wind Energy Harnessed (kWh/day) for 2025–2030:")
                    st.dataframe(future_data[['DATE', 'Total Energy (kWh/day)']].head())

                    # Visualization
                    st.subheader("Wind Speed Prediction Plot:")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(future_data['DATE'], future_data['Total Energy (kWh/day)'], label="Total Energy Harnessed")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Total Wind Speed (kWh/day)")
                    ax.set_title("Predicted Wind Speed for the year (2025–2030) for Chennai,TamilNadu")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.warning("Warning: 'Max Wind Speed (mps)' data is empty in the uploaded file. Cannot generate prediction.")

        except FileNotFoundError:
            st.error("Error: The uploaded file was not found.")
        except KeyError as e:
            st.error(f"Error: Column '{e}' not found in the uploaded file. Please ensure the file has 'DATE' and 'Max Wind Speed (mps)' columns.")
        except Exception as e:
            st.error(f"An error occurred while processing the uploaded file: {e}")

if __name__ == "__main__":
    show()
