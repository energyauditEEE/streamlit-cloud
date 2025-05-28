import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show():
    st.header("Solar Energy Dashboard")

    # Constants for installation cost calculation
    PANEL_COST_PER_WATT = {"Residential": 50, "Commercial": 45}  # ₹ per watt
    INVERTER_COST = 40_000  # ₹ (fixed cost for inverter)
    BATTERY_COST_PER_KWH = 15_000  # ₹ per kWh (optional)
    STRUCTURE_COST = 10_000  # ₹ (fixed cost for mounting structure)
    WIRING_ACCESSORIES_COST = 12_000  # ₹ (fixed cost for wiring and accessories)
    INSTALLATION_LABOR_COST = 25_000  # ₹ (fixed cost for labor)

    # Constants for panel types and electrical characteristics
    RESIDENTIAL_PANEL_AREA = 1.6  # m² (for 60-cell panels)
    COMMERCIAL_PANEL_AREA = 2.0  # m² (for 72-cell panels)
    VOLTAGE_PER_PANEL = 30  # Typical voltage for a panel in series connection
    CURRENT_PER_PANEL = 8  # Typical current for a panel in parallel connection

    st.subheader("Solar System Sizing")
    # User Inputs for System Sizing
    energy_consumption = st.number_input("Enter daily energy consumption (kWh):", min_value=1.0, value=30.0)
    sunlight_hours = st.slider("Enter peak sunlight hours per day:", min_value=3, max_value=8, value=5)
    efficiency_factor = st.slider("Enter system efficiency factor (%):", min_value=70, max_value=100, value=90) / 100
    calculated_system_size = 0.0

    calculate_button_pressed = st.button("Calculate Recommended System Size")

    if calculate_button_pressed:
        if energy_consumption > 0 and sunlight_hours > 0 and efficiency_factor > 0:
            calculated_system_size = energy_consumption / (sunlight_hours * efficiency_factor)
            st.write(f"Recommended System Size: {calculated_system_size:.2f} kW")
        else:
            st.error("Please provide valid inputs for energy consumption, sunlight hours, and efficiency.")


    st.subheader("Solar Energy Prediction")
    # File Upload Section for Dashboard
    uploaded_file = st.file_uploader("Upload your solar irradiance data Excel file with DATE, solar irradiance", type=["xlsx"])

    # Initialize data as None or an empty DataFrame outside the if uploaded_file block
    # This helps in maintaining its state across reruns if no file is uploaded yet
    data = None

    if uploaded_file:
        try:
            # Load the dataset
            data = pd.read_excel(uploaded_file, sheet_name='Sheet1')
            data.columns = data.columns.str.strip().str.lower()  # Normalize column names to lowercase
            data['date'] = pd.to_datetime(data['date'], errors='coerce')  # Convert Excel date to datetime

            if 'solar irradiance' not in data.columns:
                st.error("Error: 'solar irradiance' column not found in the uploaded file.")
                data = None # Reset data if column is missing
            else:
                # Ensure solar irradiance column is numeric and handle NaNs for mean calculation
                data['solar irradiance'] = pd.to_numeric(data['solar irradiance'], errors='coerce')
                if data['solar irradiance'].isnull().all():
                    st.error("Error: 'solar irradiance' column contains no valid numeric data.")
                    data = None # Reset data if all values are NaN
                else:
                    st.success("File uploaded and processed successfully!")

        except Exception as e:
            st.error(f"An error occurred while processing the uploaded file: {e}")
            data = None # Reset data on any processing error

    # Proceed with calculations and prediction only if data is successfully loaded
    if data is not None:
        # User Inputs for Panel Type and Total Area
        panel_type = st.radio("Select Panel Type:", options=["Residential (60-cell)", "Commercial (72-cell)"])
        total_area = st.number_input("Enter Total Panel Area (in m²):", min_value=1.0, value=10.0)

        # Set panel area based on selection
        if panel_type == "Residential (60-cell)":
            panel_area_per_unit = RESIDENTIAL_PANEL_AREA
        else:
            panel_area_per_unit = COMMERCIAL_PANEL_AREA

        # Calculate number of panels
        num_panels = int(total_area / panel_area_per_unit)
        st.write(f"Number of {panel_type} panels connected: {num_panels}")

        # User Input for Efficiency
        panel_efficiency = st.slider("Select Solar Panel Efficiency (as a percentage):", min_value=5, max_value=25, value=18) / 100

        # User input for parallel connections
        # Ensure num_parallel does not exceed num_panels
        num_parallel = st.slider("Enter the number of panels in parallel connection:", min_value=1, max_value=max(1, num_panels), value=1)

        calculate_electrical_button = st.button("Calculate Electrical Characteristics and Predict Energy")

        if calculate_electrical_button:
            if num_panels == 0:
                st.warning("Cannot calculate electrical characteristics: No panels are connected based on the total area and panel type selected.")
            else:
                # Calculate series connections
                # Ensure num_parallel is not zero to avoid ZeroDivisionError
                if num_parallel > 0:
                    num_series = num_panels // num_parallel
                else:
                    num_series = num_panels # If parallel is 0 (though slider prevents this), assume all in series for calculation

                st.write(f"Parallel Connections: {num_parallel}")
                st.write(f"Series Connections: {num_series}")

                # Electrical characteristics
                total_voltage = VOLTAGE_PER_PANEL * num_series
                total_current = CURRENT_PER_PANEL * num_parallel
                st.write(f"Total Voltage (V): {total_voltage}")
                st.write(f"Total Current (A): {total_current}")

                # Predict Solar Irradiance for 2025–2030
                future_dates = pd.date_range(start='2025-01-01', end='2030-12-31', freq='D')

                # Use the mean of 'solar irradiance' if the column is present and has valid data
                if 'solar irradiance' in data.columns and not data['solar irradiance'].isnull().all():
                    # Get actual non-null solar irradiance values to choose from
                    valid_irradiance_data = data['solar irradiance'].dropna().values
                    if len(valid_irradiance_data) > 0:
                        future_data = pd.DataFrame({
                            'date': future_dates,
                            # Randomly sample from existing valid solar irradiance data
                            'solar irradiance': np.random.choice(valid_irradiance_data, size=len(future_dates))
                        })

                        # Calculate Solar Energy Output
                        future_data['solar energy (kWh/day)'] = (
                            future_data['solar irradiance'] * panel_efficiency * total_area / 1000 # Convert Watt-hours to kWh
                        )

                        # Display Predictions
                        st.write("Predicted Solar Energy (kWh/day) for 2025–2030:")
                        st.dataframe(future_data[['date', 'solar energy (kWh/day)']].head()) # Display first few rows

                        # Visualization
                        st.write("Solar Energy Prediction Plot:")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(future_data['date'], future_data['solar energy (kWh/day)'], label="Predicted Energy")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Solar Energy (kWh/day)")
                        ax.set_title("Predicted Solar Energy (2025–2030)")
                        ax.legend()
                        st.pyplot(fig)
                    else:
                        st.warning("Warning: No valid 'solar irradiance' data points available for prediction. Please check your uploaded file.")
                else:
                    st.warning("Warning: 'solar irradiance' data is not available or contains no valid numeric values in the uploaded file. Cannot generate prediction.")

    st.markdown("---")

    st.subheader("Solar Installation Cost Calculator")
    panel_type_calc = st.radio("Select Panel Type for Cost Calculation:", options=["Residential", "Commercial"])
    include_battery = st.checkbox("Include Battery Storage?")
    battery_capacity_kwh = 0.0
    if include_battery:
        battery_capacity_kwh = st.number_input("Enter Battery Capacity (kWh):", min_value=1.0, value=5.0)

    # Ensure calculated_system_size is properly carried over or re-calculated if necessary
    # If the user clicks "Calculate Recommended System Size" multiple times,
    # or interacts with other widgets causing a rerun, calculated_system_size might revert to 0.0.
    # It's good practice to ensure this value persists or is re-derived.
    # For simplicity, we'll assume it holds its last calculated value here.
    # If not, you might need to use st.session_state to store it.

    # Re-calculate system size if the "Calculate Recommended System Size" button was not pressed
    # and the user directly navigates to the cost section or if it was cleared.
    if calculated_system_size == 0.0 and energy_consumption > 0 and sunlight_hours > 0 and efficiency_factor > 0:
         calculated_system_size = energy_consumption / (sunlight_hours * efficiency_factor)

    panel_cost = PANEL_COST_PER_WATT[panel_type_calc] * calculated_system_size * 1000 if calculated_system_size > 0 else 0
    inverter_cost = INVERTER_COST
    battery_cost = battery_capacity_kwh * BATTERY_COST_PER_KWH
    total_cost = (
        panel_cost
        + inverter_cost
        + battery_cost
        + STRUCTURE_COST
        + WIRING_ACCESSORIES_COST
        + INSTALLATION_LABOR_COST
    )

    # Cost Breakdown Display
    st.write("### Cost Breakdown:")
    st.write(f"Solar Panels: ₹{panel_cost:,.2f}")
    st.write(f"Inverter: ₹{inverter_cost:,.2f}")
    st.write(f"Battery (if included): ₹{battery_cost:,.2f}")
    st.write(f"Mounting Structure: ₹{STRUCTURE_COST:,.2f}")
    st.write(f"Wiring & Accessories: ₹{WIRING_ACCESSORIES_COST:,.2f}")
    st.write(f"Installation & Labor: ₹{INSTALLATION_LABOR_COST:,.2f}")
    st.write(f"#### Total Estimated Cost: ₹{total_cost:,.2f}")

    # Subsidy Selection
    subsidy_percent = st.slider("Select Government Subsidy (%):", min_value=0, max_value=40, value=20)
    subsidy_amount = (subsidy_percent / 100) * total_cost
    final_cost = total_cost - subsidy_amount

    # Display Final Cost
    st.write("### Final Cost After Subsidy:")
    st.write(f"Government Subsidy: ₹{subsidy_amount:,.2f}")
    st.write(f"Final Cost: ₹{final_cost:,.2f}")

if __name__ == "__main__":
    show()
