import streamlit as st
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
    
