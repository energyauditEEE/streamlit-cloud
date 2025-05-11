# main.py
import streamlit as st
import savingsplan
import anomaly
import compare
import costcalculator
import wind
import solar

# Configure page settings
st.set_page_config(
    page_title="An ML : Energy Auditor Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Energy Auditing Suite")
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Select Module", [
    "🏠 Home",
    "🔍Data Prediction",
    "📈 Forecasting",
    "📜 Energy Plans",
    "💸 Cost Calculator",
    "🌬️ Wind Energy",
    "☀️ Solar Energy"
])

# Home Page
if app_mode == "🏠 Home":
    st.title("Welcome to the Energy Analytics Suite")
 
# Module Routing

    
elif app_mode == "🔍 Data Prediction ":
    anomaly.main()
    
elif app_mode == "📈 Forecasting":
    compare.main()

elif app_mode == "📜 Energy Plans":
    savingsplan.main()

elif app_mode == "💸 Cost Calculator":
    costcalculator.main()
    
elif app_mode == "🌬️ Wind Energy":
    wind.main()
    
elif app_mode == "☀️ Solar Energy":
    solar.main()

