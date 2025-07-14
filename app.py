import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
with open('lung_cancer_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set Streamlit app title
st.markdown("""
    <h1 style="color: #2d2d2d; text-align: center;">🧠 Lung Cancer Risk Analyzer</h1>
    <p style="color: #4B4B4D; font-size: 18px; text-align: center;">Enter the information below to assess your lung cancer risk.</p>
""", unsafe_allow_html=True)

# Custom styles
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #F2F2F2;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            height: 50px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stRadio>div {
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Function to convert Yes/No to 1/0
def yes_no_to_int(answer):
    return 1 if answer == "Yes" else 0

# Collect user input
st.sidebar.header("Patient Information")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)

smoking = st.sidebar.radio("Smoking", ["No", "Yes"])
smoking_duration = st.sidebar.number_input("Smoking Duration (years)", min_value=0, value=10)
pack_years = st.sidebar.number_input("Pack Years", min_value=0, value=5)
secondhand_smoke = st.sidebar.radio("Exposure to Secondhand Smoke", ["No", "Yes"])

alcohol_consumption = st.sidebar.radio("Alcohol Consumption", ["No", "Yes"])
exercise = st.sidebar.radio("Regular Exercise (at least 30 minutes per day)", ["No", "Yes"])
diet = st.sidebar.radio("Healthy Diet (Fruits, Vegetables, Balanced Nutrition)", ["No", "Yes"])
daily_water_intake = st.sidebar.number_input("Daily Water Intake (liters)", min_value=0.0, value=2.0)

yellow_fingers = st.sidebar.radio("Yellow Fingers", ["No", "Yes"])
anxiety = st.sidebar.radio("Anxiety", ["No", "Yes"])
peer_pressure = st.sidebar.radio("Peer Pressure", ["No", "Yes"])
chronic_disease = st.sidebar.radio("Chronic Disease", ["No", "Yes"])
fatigue = st.sidebar.radio("Fatigue", ["No", "Yes"])
allergy = st.sidebar.radio("Allergy", ["No", "Yes"])
wheezing = st.sidebar.radio("Wheezing", ["No", "Yes"])
coughing = st.sidebar.radio("Coughing", ["No", "Yes"])
shortness_of_breath = st.sidebar.radio("Shortness of Breath", ["No", "Yes"])
chest_pain = st.sidebar.radio("Chest Pain", ["No", "Yes"])

previous_infections = st.sidebar.radio("Previous Lung Infections", ["No", "Yes"])
genetic_disorders = st.sidebar.radio("Genetic Disorders", ["No", "Yes"])
family_history = st.sidebar.radio("Family History of Lung Cancer", ["No", "Yes"])
pollution_exposure = st.sidebar.radio("Exposure to Air Pollution (e.g., city, factory)", ["No", "Yes"])
occupation = st.sidebar.radio("Work in High-Risk Occupation (e.g., mining, construction)", ["No", "Yes"])

bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=0.0, value=22.0)
stress_level = st.sidebar.radio("Stress Level", ["Low", "Medium", "High"])

# Gender input
gender = st.sidebar.radio("Gender", ["Male", "Female"])

# Convert the user input into 0 or 1
smoking = yes_no_to_int(smoking)
secondhand_smoke = yes_no_to_int(secondhand_smoke)
alcohol_consumption = yes_no_to_int(alcohol_consumption)
exercise = yes_no_to_int(exercise)
diet = yes_no_to_int(diet)
yellow_fingers = yes_no_to_int(yellow_fingers)
anxiety = yes_no_to_int(anxiety)
peer_pressure = yes_no_to_int(peer_pressure)
chronic_disease = yes_no_to_int(chronic_disease)
fatigue = yes_no_to_int(fatigue)
allergy = yes_no_to_int(allergy)
wheezing = yes_no_to_int(wheezing)
coughing = yes_no_to_int(coughing)
shortness_of_breath = yes_no_to_int(shortness_of_breath)
chest_pain = yes_no_to_int(chest_pain)
previous_infections = yes_no_to_int(previous_infections)
genetic_disorders = yes_no_to_int(genetic_disorders)
family_history = yes_no_to_int(family_history)
pollution_exposure = yes_no_to_int(pollution_exposure)
occupation = yes_no_to_int(occupation)

# Gender conversion: male = 1, female = 0
gender = 1 if gender == "Male" else 0

# Prepare input for prediction
user_input = np.array([[age, smoking, smoking_duration, pack_years, secondhand_smoke, alcohol_consumption,
                        exercise, diet, daily_water_intake, yellow_fingers, anxiety, peer_pressure,
                        chronic_disease, fatigue, allergy, wheezing, coughing, shortness_of_breath, chest_pain,
                        previous_infections, genetic_disorders, family_history, pollution_exposure, occupation,
                        bmi, stress_level, gender]])

# Function to predict risk
def predict_risk(input_data):
    probability = model.predict_proba(input_data)[0][1]  # probability of lung cancer
    percentage = probability * 100
    return percentage

# When button is clicked
if st.sidebar.button('Predict Lung Cancer Risk'):
    risk_percentage = predict_risk(user_input)

    # Show prediction
    st.success(f"🧠 Prediction: {risk_percentage:.2f}% chance of Lung Cancer.")

    # Risk meter (color-coded)
    if risk_percentage < 30:
        risk_label = "Low Risk"
        risk_color = "green"
    elif 30 <= risk_percentage < 70:
        risk_label = "Medium Risk"
        risk_color = "orange"
    else:
        risk_label = "High Risk"
        risk_color = "red"
        
    st.markdown(f"<h3 style='color:{risk_color};'>{risk_label}</h3>", unsafe_allow_html=True)

    # Prediction chart: risk vs non-risk
    st.subheader("Prediction Visualization")
    fig, ax = plt.subplots()
    ax.barh(["Risk", "Non-Risk"], [risk_percentage, 100 - risk_percentage], color=[risk_color, 'lightgrey'])
    ax.set_xlim(0, 100)
    ax.set_xlabel('Percentage')
    ax.set_title('Risk vs Non-Risk')

    st.pyplot(fig)

    # Add some info/insight based on the prediction
    if risk_percentage >= 70:
        st.markdown("🔴 **High Risk**: It is advised to consult a doctor immediately for further tests and preventive measures.")
    elif 30 <= risk_percentage < 70:
        st.markdown("🟠 **Medium Risk**: You may want to schedule a check-up for a thorough evaluation.")
    else:
        st.markdown("🟢 **Low Risk**: You have a low chance of lung cancer, but maintain a healthy lifestyle.")
