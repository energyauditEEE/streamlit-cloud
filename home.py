import streamlit as st
from anomaly import show as show_anomaly
from compare import show as show_compare
from savingsplan import show as show_savingsplan
from solar import show as show_solar
from wind import show as show_wind
from costcalculator import show as show_costcalculator
def home():
    st.title("An ML : Energy Auditor Dashboard")
page = st.sidebar.selectbox("Choose Module", ["Data Prediction",
                                                 "Forecasting",
                                                 "Savings Plan",
                                                 "Solar Dashboard",
                                                 "Wind Dashboard",
                                                 "Cost Calculator"])
if page == "Data Prediction":
    from anomaly import home
    home()
if page == "Forecasting":
    from compare import home
    home()
if page == "Savings Plan":
    from savingsplan import home
    home()
if page == "Solar Dashboard":
    from solar import home
    home()
if page == "Wind Dashboard":
    from wind import home
    home()
if page == "Cost Calculator":
    from costcalculator import home
    home()
    
