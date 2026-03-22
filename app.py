import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- LOAD MODEL --------------------
model, scaler = joblib.load("model.pkl")

st.set_page_config(page_title="Manufacturing Dashboard", layout="centered")

st.title("🏭 Manufacturing Output Prediction Dashboard")
st.markdown("Enter machine parameters to predict **Parts Per Hour**")

# -------------------- INPUTS --------------------
st.subheader("🔢 Machine Parameters")

col1, col2 = st.columns(2)

with col1:
    inj_temp = st.number_input("Injection Temperature", 0.0, 500.0, 200.0)
    cycle_time = st.number_input("Cycle Time", 0.0, 100.0, 30.0)
    viscosity = st.number_input("Material Viscosity", 0.0, 1000.0, 300.0)
    machine_age = st.number_input("Machine Age", 0.0, 50.0, 5.0)
    maintenance = st.number_input("Maintenance Hours", 0.0, 100.0, 10.0)

with col2:
    inj_pressure = st.number_input("Injection Pressure", 1.0, 500.0, 150.0)
    cooling_time = st.number_input("Cooling Time", 0.0, 100.0, 10.0)
    ambient_temp = st.number_input("Ambient Temperature", 0.0, 150.0, 25.0)
    operator_exp = st.number_input("Operator Experience", 0.0, 30.0, 5.0)

# -------------------- CATEGORICAL --------------------
st.subheader("📊 Machine Settings")

machine_type = st.selectbox("Machine Type", ["Type_A", "Type_B", "Type_C"])
material_grade = st.selectbox("Material Grade", ["Premium", "Standard"])
day = st.selectbox("Day of Week", 
                   ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

# -------------------- PREDICTION --------------------
if st.button("🚀 Predict Output"):

    input_df = pd.DataFrame([{
        "Injection_Temperature": inj_temp,
        "Injection_Pressure": inj_pressure,
        "Cycle_Time": cycle_time,
        "Cooling_Time": cooling_time,
        "Material_Viscosity": viscosity,
        "Ambient_Temperature": ambient_temp,
        "Machine_Age": machine_age,
        "Operator_Experience": operator_exp,
        "Maintenance_Hours": maintenance,
        "Machine_Type": machine_type,
        "Material_Grade": material_grade,
        "Day_of_Week": day
    }])

    # Feature Engineering
    input_df["Temperature_Pressure_Ratio"] = (
        input_df["Injection_Temperature"] / (input_df["Injection_Pressure"] + 1e-5)
    )

    # Encoding
    input_df = pd.get_dummies(input_df)

    # Align features
    model_features = scaler.feature_names_in_
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Handle negative
    prediction = max(0, prediction)

    # -------------------- OUTPUT --------------------
    st.subheader("📈 Prediction Result")
    st.success(f"Predicted Parts Per Hour: {round(prediction, 2)}")

    # -------------------- PERFORMANCE LABEL --------------------
    if prediction > 40:
        st.success("🔥 High Efficiency")
    elif prediction > 25:
        st.warning("⚠️ Medium Efficiency")
    else:
        st.error("❌ Low Efficiency")

    # -------------------- INPUT VISUALIZATION --------------------
    st.subheader("📊 Input Parameter Overview")

    input_values = [
        inj_temp, inj_pressure, cycle_time, cooling_time,
        viscosity, ambient_temp, machine_age, operator_exp, maintenance
    ]

    labels = [
        "Temp", "Pressure", "Cycle", "Cooling",
        "Viscosity", "Ambient", "Age", "Experience", "Maintenance"
    ]

    plt.figure()
    plt.bar(labels, input_values)
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # -------------------- FEATURE IMPORTANCE --------------------
    if hasattr(model, "feature_importances_"):
        st.subheader("⭐ Top Influencing Factors")

        importances = model.feature_importances_
        feat_names = scaler.feature_names_in_

        feat_df = pd.DataFrame({
            "Feature": feat_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(5)

        st.write(feat_df)

        plt.figure()
        plt.barh(feat_df["Feature"], feat_df["Importance"])
        plt.gca().invert_yaxis()
        st.pyplot(plt)