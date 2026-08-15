import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title = "SmartPremimum",layout="wide")

@st.cache_resource
def load_artifacts():
    with open("model.pkl","rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl","rb") as f:
        encoders= pickle.load(f)
    with open("feature_columns.pkl","rb") as f:
        feature_columns = pickle.load(f)

    return model,encoders,feature_columns

model,encoders,feature_columns = load_artifacts()

st.title("SmartPremimum")
st.markdown("Insurance Premimum Prediction System")

st.write("  Enter the customer, health, insurance and property details to estimate the insurance premium.")

st.header("Customer Information")
col1,col2,col3 = st.columns(3)

with col1:
    age = st.number_input("Age",min_value=18,max_value=85,value=30,step=1)
    gender = st.selectbox("Gender",["Male","Female"])
    annual_income = st.number_input("Annual_income",min_value=0.0,value=50000.0,step=1000.0)
with col2:
    marital_status = st.selectbox("Marital Status",["Single","Married","Divorced"])
    dependents = st.number_input("Dependents",min_value=0,max_value=20,step=1,value=1)
    education_level = st.selectbox("Education Level",["High School","Bachelor's","Master's","PhD"])
with col3:
    occupation = st.selectbox("Occupation",["Employed","Self_Unemployed","Unemployed"])
    location = st.selectbox("Location",["Urban","Suburban","Rural"])

st.header("Health & Lifestyle")
col1,col2,col3 = st.columns(3)

with col1:
    health_score = st.number_input("Health_score",min_value=0.0,max_value=100.0,value=50.0,step=0.1)
    previous_claims = st.number_input("Previous Calims",min_value=0,max_value=50,value=0,step=1)
with col2:
    credit_score = st.number_input("Credit Score",min_value=300.0,max_value=850.0,value=650.0,step=1.0)
    smoking_status = st.selectbox("Smoking Status",["Yes","No"])
with col3:
    exercise_frequency = st.selectbox("Exercise_frequency",["Daily","Weekly","Monthly","Rarely"])
    customer_feedback = st.selectbox("Customer_feedback",["Good","Average","Poor"])

st.header("Insurance Information")
col1,col2,col3 = st.columns(3)

with col1:
    policy_type = st.selectbox("Policy_type",["Basic","Comprehensive","Premimum"])
with col2:
    insurance_duration = st.number_input("Insurance Duration",min_value=0,max_value=50,value=5,step=1)
with col3:
    vehicle_age = st.number_input("Vehicle_age",min_value=0.0,max_value=50.0,value=5.0,step=0.1)

st.header("Property Information")
property_type = st.selectbox("Property_type",["House","Apartment","Condo"])

st.header("Policy Information")
policy_date = st.date_input("Policy Start Date")
policy_time = st.time_input("Policy Start Time")

def create_input_dataframe():
    policy_datetime = pd.Timestamp(
        year=policy_date.year,
        month=policy_date.month,
        day=policy_date.day,
        hour=policy_time.hour,
        minute=policy_time.minute,
        second=0
    )

    data = pd.DataFrame({
        "Age": [age],
        "Annual Income": [annual_income],
        "Gender": [gender],
        "Marital Status": [marital_status],
        "Number of Dependents": [dependents],
        "Education Level": [education_level],
        "Occupation": [occupation],
        "Health Score": [health_score],
        "Location": [location],
        "Policy Type": [policy_type],
        "Previous Claims": [previous_claims],
        "Vehicle Age": [vehicle_age],
        "Credit Score": [credit_score],
        "Insurance Duration": [insurance_duration],
        "Customer Feedback": [customer_feedback],
        "Smoking Status": [smoking_status],
        "Exercise Frequency": [exercise_frequency],
        "Property Type": [property_type],
        "Year": [policy_datetime.year],
        "Month": [policy_datetime.month],
        "Day": [policy_datetime.day],
        "Dayofyear": [policy_datetime.day_of_year],
        "Num_of_week": [policy_datetime.day_of_week],
        "Hour": [policy_datetime.hour],
        "Minute": [policy_datetime.minute]
    })

    return data

def engineering_features(data):
    data = data.copy()

    data["income_per_dependent"] = (data["Annual Income"]/ (data["Number of Dependents"] + 1))
    data["claims_per_year"] = (data["Previous Claims"]/ (data["Insurance Duration"] + 1))
    data["income_age"] = (data["Annual Income"]* data["Age"])
    data["credit_income"] = (data["Credit Score"]* data["Annual Income"])
    data["health_claim"] = (data["Health Score"]* data["Previous Claims"])
    data["vehicle_duration"] = (data["Vehicle Age"]/ (data["Insurance Duration"] + 1))

    return data

def target_encode(data):
    data = data.copy()
    categorical_columns = [
        "Gender",
        "Marital Status",
        "Education Level",
        "Occupation",
        "Location",
        "Policy Type",
        "Customer Feedback",
        "Smoking Status",
        "Exercise Frequency",
        "Property Type"
    ]

    for col in categorical_columns:
        encoded_column = col + "_te"
        transformed = encoders[col].transform(data[[col]])
        data[encoded_column] = transformed[col]

    data.drop(columns=categorical_columns,inplace=True)

    return data

st.divider()
if st.button("Predict premimum",type="primary",use_container_width=True):
    try:
        input_data = create_input_dataframe()
        input_data = engineering_features(input_data)
        input_data = target_encode(input_data)
        input_data.replace([np.inf,-np.inf],np.nan,inplace=True)
        input_data.fillna(0,inplace=True)
        input_data = input_data.reindex(columns=feature_columns,fill_value=0)
        prediction_log = model.predict(input_data)
        prediction_actual = np.expm1(prediction_log[0])
        prediction_actual = max(0,prediction_actual)

        st.success("Premimum prediction generated successfully")
        st.metric( "Estimated Insurance Premium", f"₹ {prediction_actual:,.2f}")
        st.info("Prediction generated using the trained ""SmartPremium LightGBM model.")

    except Exception as e:
        st.error("An error occurred while generating the prediction")
        st.exception(e)

st.divider()
st.caption(
    "SmartPremium | LightGBM | 5-Fold OOF Target Encoding | "
    "MLflow Tracked Model"
)
