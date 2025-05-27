import streamlit as st
from anomaly import show as show_anomaly
from compare import show as show_compare
from savingsplan import show as show_savingsplan
from solar import show as show_solar
from wind import show as show_wind
from costcalculator import show as show_costcalculator

def main():
    st.title("Energy and Anomaly Management Dashboard")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Data Prediction",
        "Forecasting",
        "Savings Plan",
        "Solar Dashboard",
        "Wind Dashboard",
        "Cost Calculator"
    ])
    with tab1:
        show_anomaly()
    with tab2:
        show_compare()
    with tab3:
        show_savingsplan()
    with tab4:
        show_solar()
    with tab5:
        show_wind()
    with tab6:
        show_costcalculator()

if _name_ == "_main_":
    main() # Corrected: Call the main() function
