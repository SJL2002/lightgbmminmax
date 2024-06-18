import streamlit as st
import pandas as pd
import joblib
import os

# Load the RandomForest model using joblib
model_path = 'models/rfmodel.pkl'  # Adjust path as necessary
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    model = joblib.load(model_path)

# Function to preprocess data and predict
def preprocess_and_predict(gender, age, hypertension, heart_disease, ever_married, work_type, 
                           residence_type, avg_glucose_level, bmi, smoking_status):
    # Create input DataFrame for prediction
    data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })

    # Convert categorical variables to numeric using one-hot encoding
    data = pd.get_dummies(data)
    
    # Select only the features used during model training
    features_used = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                     'work_type_Private', 'work_type_Self-employed', 'work_type_Govt_job', 'work_type_children',
                     'Residence_type_Urban', 'Residence_type_Rural', 'avg_glucose_level', 'bmi',
                     'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']

    # Ensure all features used in training are present in the input data
    for feature in features_used:
        if feature not in data.columns:
            data[feature] = 0

    data = data[features_used]
    
    # Make prediction
    prediction = model.predict(data)
    return prediction[0]

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = 'input'
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

def main():
    # Page: Input
    if st.session_state.page == 'input':
        st.title('Stroke Prediction')
        st.write('This app predicts the likelihood of a stroke based on input features.')

        # Input fields for user input
        gender = st.selectbox('Gender', ['Male', 'Female'])
        age = st.slider('Age', 0, 150, 50)
        hypertension = st.checkbox('Hypertension')
        heart_disease = st.checkbox('Heart Disease')
        ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
        work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children'])
        residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
        avg_glucose_level = st.number_input('Average Glucose Level', min_value=50.0, max_value=500.0, value=100.0, step=0.1)
        bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes'])
        
        # Convert categorical inputs to binary/numeric as needed
        gender_num = 1 if gender == 'Male' else 0
        hypertension_num = 1 if hypertension else 0
        heart_disease_num = 1 if heart_disease else 0
        ever_married_num = 1 if ever_married == 'Yes' else 0
        residence_num = 1 if residence_type == 'Urban' else 0

        # Predict stroke probability when 'Predict' button is clicked
        if st.button('Predict'):
            prediction = preprocess_and_predict(gender_num, age, hypertension_num, heart_disease_num, ever_married_num, 
                                                work_type, residence_num, avg_glucose_level, bmi, smoking_status)
            st.session_state.prediction = prediction
            st.session_state.page = 'result'

    # Page: Result
    elif st.session_state.page == 'result':
        st.header('Prediction Result')
        st.write(f'Prediction: {st.session_state.prediction}')  # Display prediction result
        
        if st.button('Go back to input'):
            st.session_state.page = 'input'

# Run the app
if __name__ == '__main__':
    main()
