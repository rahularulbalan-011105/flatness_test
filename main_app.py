import streamlit as st
import pandas as pd
import joblib

# Load models
rf_model = joblib.load("rf_flatness_model.pkl")
xgb_model = joblib.load("xgb_flatness_model.pkl")

st.title("🧾 Flatness Prediction System")

st.write("Enter the rolling parameters to predict the Flatness I-Units")

# Input fields
entry_thickness = st.number_input("Entry Thickness (mm)", 0.0, 10.0, 2.0)
exit_thickness = st.number_input("Exit Thickness (mm)", 0.0, 10.0, 1.8)
rolling_speed = st.number_input("Rolling Speed (m/s)", 0.0, 10.0, 4.0)
roll_bending_force = st.number_input("Roll Bending Force (kN)", 0.0, 500.0, 250.0)
tension = st.number_input("Tension (kN)", 0.0, 200.0, 100.0)
strip_temp = st.number_input("Strip Temperature (°C)", 0.0, 800.0, 400.0)
lubrication = st.number_input("Lubrication Coefficient", 0.0, 1.0, 0.3)

model_choice = st.selectbox("Choose Model", ["Random Forest", "XGBoost"])

if st.button("🔍 Predict Flatness"):
    # Prepare input
    input_data = pd.DataFrame({
        'Entry_Thickness_mm': [entry_thickness],
        'Exit_Thickness_mm': [exit_thickness],
        'Rolling_Speed_mps': [rolling_speed],
        'Roll_Bending_Force_kN': [roll_bending_force],
        'Tension_kN': [tension],
        'Strip_Temperature_C': [strip_temp],
        'Lubrication_Coeff': [lubrication]
    })

    # Choose model
    model = rf_model if model_choice == "Random Forest" else xgb_model

    # Predict
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Flatness I-Units: {prediction:.2f}")

    # Decision suggestion
    if prediction < 5:
        action = "✅ Flatness is good. No correction needed."
    elif 5 <= prediction < 10:
        action = "⚙️ Apply minor roll bending or adjust lubrication."
    elif 10 <= prediction < 15:
        action = "🔧 Perform thermal equalization or mild tension correction."
    else:
        action = "🔥 Severe issue. Perform surface flattening or annealing."

    st.info(f"Recommended Action: {action}")
