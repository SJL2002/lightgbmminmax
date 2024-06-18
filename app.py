import streamlit as st
import pandas as pd
import joblib
import os

# Load the RandomForest model using joblib
model_path = 'models/rfmodel.pkl'
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    model = joblib.load(model_path)

# Function to preprocess data and predict
def preprocess_and_predict(input_data):
    # Select only the features used during model training
    features_used = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                     'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
    
    # Convert categorical variables to numeric using one-hot encoding
    input_data = pd.get_dummies(input_data)
    
    # Align input data columns with the model's expected features
    missing_cols = set(features_used) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[features_used]
    
    # Make prediction
    prediction = model.predict(input_data)
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

        # Upload data file
        uploaded_file = st.file_uploader("Choose a CSV file...", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            
            if st.button('Predict'):
                if all(feature in data.columns for feature in features_used):
                    # Predict for all rows in the uploaded file
                    data['Prediction'] = data.apply(preprocess_and_predict, axis=1)
                    st.session_state.prediction = data
                    st.session_state.page = 'result'
                else:
                    st.error("Uploaded file does not contain the required features.")

    # Page: Result
    elif st.session_state.page == 'result':
        st.header('Prediction Result')
        st.write(st.session_state.prediction)  # Display the predictions
        
        st.write("Was this result correct?")
        if st.button('Yes'):
            st.session_state.page = 'feedback'
        if st.button('No'):
            st.session_state.page = 'feedback'

    # Page: Feedback
    elif st.session_state.page == 'feedback':
        st.header('Feedback')
        st.write("Thank you for your feedback!")
        st.write("You can go back to input to make another prediction.")
        if st.button('Go back to input'):
            st.session_state.page = 'input'

# Run the app
if __name__ == '__main__':
    main()
