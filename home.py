import streamlit as st
def main():
    st.title("An ML : Energy Auditor Dashboard")
page = st.sidebar.selectbox("Choose Module", ["Data Prediction",
                                                 "Forecasting",
                                                 "Savings Plan",
                                                 "Solar Dashboard",
                                                 "Wind Dashboard",
                                                 "Cost Calculator"])
if page == "Data Prediction":
    from anomaly import main
    main()
if page == "Forecasting":
    from compare import main
    main()
if page == "Savings Plan":
    from savingsplan import main
    main()
if page == "Solar Dashboard":
    from solar import main
    main()
if page == "Wind Dashboard":
    from wind import main
    main()
if page == "Cost Calculator":
    from costcalculator import main
    main()
    
