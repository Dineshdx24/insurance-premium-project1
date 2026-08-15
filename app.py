import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="SmartPremium",layout="wide")

@st.cache_resource
def load_pipeline():
    final_pipeline = joblib.load("model_artifacts/final_pipeline.pkl")
    return final_pipeline

final_pipeline = load_pipeline()

st.title("SmartPremium")
st.write("Insurance Premium Prediction System")
st.write(
    "Enter the customer, health, insurance and property "
    "details to estimate the insurance premium."
)

st.header("Customer Information")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age",min_value=18,max_value=100,value=30,step=1)
    gender = st.selectbox("Gender",["Male", "Female"])
    annual_income = st.number_input("Annual Income",min_value=0.0,value=50000.0,step=1000.0)
with col2:
    marital_status = st.selectbox("Marital Status",["Single", "Married", "Divorced"])
    dependents = st.number_input("Number of Dependents",min_value=0.0,max_value=20.0,value=1.0,step=1.0)
    education_level = st.selectbox("Education Level",["High School","Bachelor's","Master's","PhD"])
with col3:
    occupation = st.selectbox("Occupation",["Employed","Self-Employed","Unemployed"])
    location = st.selectbox("Location",["Urban","Suburban","Rural"])

st.header("Health & Lifestyle")

col1, col2, col3 = st.columns(3)

with col1:
    health_score = st.number_input("Health Score",min_value=0.0,max_value=100.0,value=50.0,step=0.1)
    previous_claims = st.number_input("Previous Claims",min_value=0.0,max_value=50.0,value=0.0,step=1.0)
with col2:
    credit_score = st.number_input("Credit Score",min_value=0.0,max_value=900.0,value=650.0,step=1.0)
    smoking_status = st.selectbox("Smoking Status",["Yes", "No"])
with col3:
    exercise_frequency = st.selectbox("Exercise Frequency",["Daily","Weekly","Monthly","Rarely"])
    customer_feedback = st.selectbox("Customer Feedback",["Good","Average","Poor"])

st.header("Insurance Information")

col1, col2, col3 = st.columns(3)

with col1:
    policy_type = st.selectbox("Policy Type",["Basic","Comprehensive","Premium"])
with col2:
    insurance_duration = st.number_input("Insurance Duration",min_value=1.0,max_value=50.0,value=5.0,step=1.0)
with col3:
    vehicle_age = st.number_input("Vehicle Age",min_value=0.0,max_value=50.0,value=5.0,step=1.0)

st.header("Property Information")

property_type = st.selectbox("Property Type",["House","Apartment","Condo"])

st.header("Policy Information")

col1, col2 = st.columns(2)

with col1:
    policy_start_date = st.date_input("Policy Start Date",value=datetime.today())
with col2:
    policy_start_time = st.time_input("Policy Start Time",value=datetime.now().time())

if st.button("Predict Premium",type="primary",use_container_width=True):
    try:
        policy_datetime = pd.Timestamp.combine(policy_start_date,policy_start_time)
    
        customer_data = pd.DataFrame({

            "Age": [float(age)],
            "Gender": [gender],
            "Annual Income": [float(annual_income)],
            "Marital Status": [marital_status],
            "Number of Dependents": [float(dependents)],
            "Education Level": [education_level],
            "Occupation": [occupation],
            "Health Score": [float(health_score)],
            "Location": [location],
            "Policy Type": [policy_type],
            "Previous Claims": [float(previous_claims)],
            "Vehicle Age": [float(vehicle_age)],
            "Credit Score": [float(credit_score)],
            "Insurance Duration": [float(insurance_duration)],
            "Customer Feedback": [customer_feedback],
            "Smoking Status": [smoking_status],
            "Exercise Frequency": [exercise_frequency],
            "Property Type": [property_type],
            "Policy Start Date": [policy_datetime]
        })

        prediction = final_pipeline.predict(customer_data)

        premium = float(prediction[0])

        premium = max(0, premium)

        st.success("Premium prediction generated successfully!")
        st.subheader(f"Estimated Insurance Premium: ₹{premium:,.2f}")
        st.info(
            "Prediction generated using the SmartPremium "
            "end-to-end machine learning pipeline."
        )

    except Exception as e:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.exception(e)
