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
def predict_missing(data):
    """Predicts missing power values with safety checks"""
    if data is None or 'Power_Consumption(MU)' not in data.columns:
        return data
        
    known = data[data['Power_Consumption(MU)'].notna()]
    missing = data[data['Power_Consumption(MU)'].isna()]
    
    if len(known) < 10:  # Minimum samples check
        st.warning("Insufficient data for reliable prediction")
        return data
        
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        features = ['Temperature (F)', 'Dew Point (F)', 
                   'Max Wind Speed (mps)', 'Avg Wind Speed (mps)',
                   'Atm Pressure (hPa)', 'Humidity(g/m^3)']
                   
        model.fit(known[features], known['Power_Consumption(MU)'])
        missing['Power_Consumption(MU)'] = model.predict(missing[features])
        
        return pd.concat([known, missing]).sort_values('DATE')
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return data

@st.cache_data
def detect_anomalies(data):
    """Robust anomaly detection with fallbacks"""
    if data is None or 'Power_Consumption(MU)' not in data.columns:
        return pd.DataFrame()
        
    try:
        model = IsolationForest(contamination=0.05, random_state=42)
        data['anomaly'] = model.fit_predict(data[['Power_Consumption(MU)']])
        return data[data['anomaly'] == -1]
        
    except Exception as e:
        st.error(f"Anomaly detection failed: {str(e)}")
        return pd.DataFrame()

def show():
    """Main display function for Streamlit tab"""
    st.header("🔍 Power Consumption Diagnosis")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload Energy Data (Excel)",
        type=["xlsx"],
        help="Requires DATE, weather parameters, and Power_Consumption(MU) columns"
    )
    
    # Data processing pipeline
    raw_data = load_data(uploaded_file)
    processed_data = predict_missing(raw_data)
    anomalies = detect_anomalies(processed_data)
    
    if processed_data is not None:
        # Visualization Section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Temporal Analysis")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Main time series plot
            ax.plot(processed_data['DATE'], 
                   processed_data['Power_Consumption(MU)'], 
                   label='Consumption', zorder=1)
                   
            # Anomaly highlights
            if not anomalies.empty:
                ax.scatter(anomalies['DATE'], 
                          anomalies['Power_Consumption(MU)'],
                          color='red', label='Anomalies', zorder=2)
                          
            ax.set_xlabel("Date")
            ax.set_ylabel("Power Consumption (MU)")
            ax.legend()
            st.pyplot(fig)
            
        with col2:
            st.subheader("Data Summary")
            st.metric("Total Records", len(processed_data))
            st.metric("Anomalies Detected", len(anomalies))
            
            # Quick stats table
            stats = processed_data['Power_Consumption(MU)'].describe()
            st.dataframe(
                stats.rename('Statistics').to_frame().T,
                use_container_width=True
            )
            
        # Heatmap section
        st.subheader("Monthly Consumption Patterns")
        try:
            heatmap_data = processed_data.set_index('DATE').resample('M').mean()
            fig, ax = plt.subplots(figsize=(14, 4))
            sns.heatmap(
                heatmap_data[['Power_Consumption(MU)']].T,
                annot=True, fmt=".1f",
                cmap="YlGnBu",
                cbar_kws={'label': 'MU'}
            )
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Heatmap generation failed: {str(e)}")
            
        # Data download
        st.download_button(
            label="Download Processed Data",
            data=processed_data.to_csv(index=False),
            file_name="processed_energy_data.csv",
            mime="text/csv"
        )

# For local testing (remove in deployment)
if __name__ == "__main__":
    show()
