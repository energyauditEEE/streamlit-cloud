import streamlit as st

def show():
    st.header("Energy Saving Plan Dashboard")
    st.write("Explore energy-saving guidelines for different sectors.")

    # Enhanced Energy-saving plans with industry benchmarks
    energy_plans = {
        "Education": [
            "Turn off lights and fans after class",
            "Use LED lighting in classrooms and halls",
            "Maximize use of natural daylight",
            "Maintain electrical equipment regularly",
            "Install solar panels if possible",
            "Encourage energy-saving habits",
            "Use motion-sensor lights in restrooms and corridors",
            "Switch off computers and projectors after use",
            "Use fans and ventilation instead of ACs when possible",
            "Create an energy monitoring team",
        ],
        "Industry": {
            "Cement": [
                "Maintain air ratio for kilns (Recommended: 1.15–1.18)",
                "Optimize boiler efficiency (Target temperature: 130°C at final stage)",
                "Use waste heat recovery systems for preheating raw materials",
                "Enhance fuel combustion efficiency with automatic air-fuel control",
            ],
            "Iron & Steel": [
                "Adopt waste heat recovery for flue gases (Min 52% efficiency)",
                "Optimize power factor correction (Target > 0.95)",
                "Upgrade furnace insulation (Maintain ceiling temperature: 120°C)",
                "Use oxygen-enriched combustion systems to improve fuel utilization",
            ],
            "Textile": [
                "Reduce lighting power density (Recommended: 7.1–14.1 W/m²)",
                "Install high-efficiency IE3 motors (Min efficiency: 93%)",
                "Implement optimized compressed air systems (Leakage < 10%)",
                "Use solar-powered heating systems for drying and processing",
            ]
        },
        "Residential": {
            "Apartments": [
                "Switch to LED lights and follow lighting power density standards.",
                "Use energy-efficient elevators or encourage stair use for lower floors.",
                "Install solar panels for shared electricity supply.",
                "Optimize air-conditioning based on seasonal variations.",
                "Ensure power factor correction in common areas.",
            ],
            "Houses": [
                "Use solar water heaters instead of electric geysers.",
                "Regulate air-conditioning between 24–26°C and ensure filter maintenance.",
                "Improve thermal insulation to reduce heat loss.",
                "Use energy-efficient fans, refrigerators, and washing machines.",
                "Seal windows and doors to enhance energy efficiency.",
            ],
            "Neighborhoods": [
                "Install solar-powered street lights.",
                "Encourage carpooling and cycling for reduced emissions.",
                "Implement monthly energy audits and awareness workshops.",
                "Promote use of efficient power distribution transformers.",
                "Develop community-level waste heat recovery programs.",
            ]
        }
    }

    # --- Education Section ---
    with st.expander("🎓 Education"):
        for plan in energy_plans["Education"]:
            st.markdown(f"- {plan}")

    # --- Industry Section with EC Guidelines ---
    with st.expander("🏭 Industry"):
        industry_types = list(energy_plans["Industry"].keys())
        selected_industry = st.selectbox("Select Industry Type:", industry_types, key="industry_selectbox")
        if selected_industry:
            st.subheader(selected_industry)
            for plan in energy_plans["Industry"][selected_industry]:
                st.markdown(f"- {plan}")

    # --- Residential Section with Conservation Guidelines ---
    with st.expander("🏠 Residential"):
        residential_types = list(energy_plans["Residential"].keys())
        selected_residential = st.selectbox("Select Residential Type:", residential_types, key="residential_selectbox")
        if selected_residential:
            st.subheader(selected_residential)
            for plan in energy_plans["Residential"][selected_residential]:
                st.markdown(f"- {plan}")

    st.info("This interactive dashboard now integrates energy benchmarks from official guidelines.")

if __name__ == "__main__":
    show()