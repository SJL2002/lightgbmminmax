import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open('lgbmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Define mappings for categorical variables
gender_map = {'Female': 0, 'Male': 1}
hypertension_map = {'No': 0, 'Yes': 1}
heart_disease_map = {'No': 0, 'Yes': 1}
ever_married_map = {'No': 0, 'Yes': 1}
work_type_map = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'Children': 3}
residence_type_map = {'Urban': 0, 'Rural': 1}
smoking_status_map = {'Never_smoked': 0, 'Formerly_smoked': 1, 'Smoked': 2}

# Function to predict using the model
def predict(features):
    return model.predict_proba([features])[0][1]

# Input function
def user_input_features():
    gender = st.sidebar.selectbox('Gender', list(gender_map.keys()))
    age = st.sidebar.slider('Age', 0, 100, 50)
    hypertension = st.sidebar.selectbox('Hypertension', list(hypertension_map.keys()))
    heart_disease = st.sidebar.selectbox('Heart Disease', list(heart_disease_map.keys()))
    ever_married = st.sidebar.selectbox('Ever Married', list(ever_married_map.keys()))
    work_type = st.sidebar.selectbox('Work Type', list(work_type_map.keys()))
    residence_type = st.sidebar.selectbox('Residence Type', list(residence_type_map.keys()))
    avg_glucose_level = st.sidebar.slider('Average Glucose Level', 0.0, 300.0, 100.0)
    bmi = st.sidebar.slider('BMI', 0.0, 50.0, 25.0)
    smoking_status = st.sidebar.selectbox('Smoking Status', list(smoking_status_map.keys()))

    data = {
        'gender': gender_map[gender],
        'age': age,
        'hypertension': hypertension_map[hypertension],
        'heart_disease': heart_disease_map[heart_disease],
        'ever_married': ever_married_map[ever_married],
        'work_type': work_type_map[work_type],
        'residence_type': residence_type_map[residence_type],
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status_map[smoking_status]
    }
    
    features = list(data.values())
    return np.array(features)

st.title('Health Prediction App')

# Page logic
if 'page' not in st.session_state:
    st.session_state.page = 'input'

# Input page
if st.session_state.page == 'input':
    st.header('Input Patient Information')

    features = user_input_features()

    if st.button('Predict'):
        prediction = predict(features)
        st.session_state.prediction = prediction
        st.session_state.page = 'result'

# Result page
if st.session_state.page == 'result':
    st.header('Prediction Result')
    st.write(f'The probability of the event is: {st.session_state.prediction:.2f}')

    st.write('Was this prediction accurate?')
    if st.button('Yes'):
        st.session_state.page = 'feedback'
    if st.button('No'):
        st.session_state.page = 'feedback'

# Feedback page
if st.session_state.page == 'feedback':
    st.header('Thank you for your feedback!')
    if st.button('Go back to input'):
        st.session_state.page = 'input'
