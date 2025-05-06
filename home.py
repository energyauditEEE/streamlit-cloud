import streamlit as st
from anomaly import show as show_anomaly
from compare import show as show_compare
from costcalculator import show as show_costcalculator
from savingsplan import show as show_savingsplan
from solar import show as show_solar
from wind import show as show_wind

def main():
    st.title("Energy and Anomaly Management Dashboard")

    # Create tabs for each of your sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Anomaly Detection",
                                                 "Comparison",
                                                 "Cost Calculator",
                                                 "Savings Plan",
                                                 "Solar Dashboard",
                                                 "Wind Dashboard"])
    # Assign the functions from each file to the corresponding tab
    with tab1:
        show_anomaly()

    with tab2:
        show_compare()

    with tab3:
        show_costcalculator()

    with tab4:
        show_savingsplan()

    with tab5:
        show_solar()

    with tab6:
        show_wind()

if __name__ == "__main__":
    main()