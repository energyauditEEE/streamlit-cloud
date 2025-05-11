# main.py
import streamlit as st
import SAVINPLAN
import ANOMLY
import COMPARE
import cost
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
    ANOMLY.main()
    
elif app_mode == "📈 Forecasting":
    COMPARE.main()

elif app_mode == "📜 Energy Plans":
    SAVINPLAN.main()

elif app_mode == "💸 Cost Calculator":
    cost.main()
    
elif app_mode == "🌬️ Wind Energy":
    wind.main()
    
elif app_mode == "☀️ Solar Energy":
    solar.main()

