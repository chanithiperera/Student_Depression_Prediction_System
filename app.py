import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- 1. CONFIGURATION AND MODEL LOADING ---

# MODEL PATH 
MODEL_PATH = "random_forest_best_balanced_model.pkl" 

# FINAL CORRECT FEATURE LIST: Using the exact order provided by the user.
FEATURE_COLUMNS = [
    # 1. Binary Features (MUST use the full, long names)
    'Have_you_ever_had_suicidal_thoughts__Encoded',  
    'Academic Pressure', 
    'Financial Stress',
    'Age', 
    'Work/Study Hours',
    'Dietary Habits_Unhealthy',
    'Study Satisfaction',
    "Sleep Duration_'More than 8 hours'",
    "Sleep Duration_'Less than 5 hours'",
    'Family_History_of_Mental_Illness_Encoded', # Moved to the end as per the new list
]

# Use caching to load the model only once
@st.cache_resource
def load_model():
    """Loads the best tuned Random Forest model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"FATAL ERROR: Model file not found at {MODEL_PATH}")
        st.warning("Ensure you ran joblib.dump(your_model, 'random_forest_best_balanced_model.pkl') in your notebook.")
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- 2. STREAMLIT APP LAYOUT ---

st.set_page_config(page_title="Depression Risk Predictor", layout="wide")
st.title("🧠 Student Depression Risk Predictor")
st.markdown("Enter the student's characteristics below to get a prediction on their depression risk.")

if model is not None:
    
    # --- Input Widgets for the 10 Features ---
    st.header("1. Student Characteristics")
    col1, col2, col3 = st.columns(3)

    # Column 1: Core Numerical Inputs
    with col1:
        age = st.number_input("Age (Years)", min_value=18, max_value=60, value=22)
        academic_pressure = st.selectbox("Academic Pressure (1=Low, 5=High)", options=list(range(1, 6)), index=3)
        study_satisfaction = st.selectbox("Study Satisfaction (1=Low, 5=High)", options=list(range(1, 6)), index=1)
        financial_stress = st.selectbox("Financial Stress (1=Low, 5=High)", options=list(range(1, 6)), index=3)

    # Column 2: Work/Sleep/Diet Inputs
    with col2:
        work_hours = st.number_input("Work/Study Hours", min_value=1, max_value=15, value=6)
        unhealthy_diet = st.selectbox("Unhealthy Diet (1=Yes, 0=No)", options=[0, 1])
        sleep_less_5h = st.selectbox("Sleep < 5 hours (1=Yes, 0=No)", options=[0, 1])
        sleep_more_8h = st.selectbox("Sleep > 8 hours (1=Yes, 0=No)", options=[0, 1])

    # Column 3: Binary/Encoded Inputs
    with col3:
        # Note: Input variables remain simple for the UI logic
        suicidal_thoughts = st.selectbox("Suicidal Thoughts (1=Yes, 0=No)", options=[0, 1])
        family_history = st.selectbox("Family History (1=Yes, 0=No)", options=[0, 1])


    # --- 3. PREDICTION LOGIC ---

    if st.button("Predict Depression Risk", type="primary"):
        
        # 3.1. Collect inputs into a dictionary, ensuring keys match the full names.
        # This dictionary is assembled in the order of FEATURE_COLUMNS above.
        input_data = {
            # --- FEATURE_COLUMNS ORDER: 1st Half ---
            'Have_you_ever_had_suicidal_thoughts__Encoded': suicidal_thoughts, 
            'Academic Pressure': academic_pressure,
            'Financial Stress': financial_stress,
            'Age': age,
            'Work/Study Hours': work_hours,
            'Dietary Habits_Unhealthy': unhealthy_diet,
            'Study Satisfaction': study_satisfaction,
            
            # --- FEATURE_COLUMNS ORDER: 2nd Half ---
            "Sleep Duration_'More than 8 hours'": sleep_more_8h,
            "Sleep Duration_'Less than 5 hours'": sleep_less_5h,
            'Family_History_of_Mental_Illness_Encoded': family_history,
        }
        
        # 3.2. Convert to DataFrame, using FEATURE_COLUMNS to guarantee order
        # This line is the final guarantee of correct column names and order.
        input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]

        # 3.3. Predict
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1] # Prob of Class 1 (Depression)
        
        # 3.4. Display Result
        st.header("--- Prediction Result ---")
        
        if prediction == 1:
            st.error(f"The model predicts: **HIGH RISK OF DEPRESSION**")
            st.markdown(f"**Probability of Depression (Risk Score):** **{prediction_proba:.2f}**")
        else:
            st.success(f"The model predicts: **LOW RISK OF DEPRESSION**")
            st.markdown(f"**Probability of Depression (Risk Score):** **{prediction_proba:.2f}**")
