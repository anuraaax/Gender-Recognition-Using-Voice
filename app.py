import streamlit as st
import pandas as pd
import numpy as np
import pickle
import StandardScaler

@st.cache(allow_output_mutation=True)
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to preprocess input data
def preprocess_data(data, scaler):
    # Assuming your data is in a DataFrame format
    X = data.values
    X = scaler.transform(X)
    return X

def main():
    st.title('Gender Recognition Using Voice')

    # File upload widget
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Display uploaded data
        st.subheader('Uploaded data:')
        st.write(data)

        # Load the model and scaler
        model = load_model('gender_recognition_model.pkl')
        scaler = StandardScaler()  # Assuming you used StandardScaler during training

        # Preprocess the data
        X = preprocess_data(data, scaler)

        # Make predictions
        predictions = model.predict(X)
        genders = ["Female" if pred == 0 else "Male" for pred in predictions]

        # Display predictions
        st.subheader('Predicted Genders:')
        st.write(", ".join(genders))

if __name__ == '__main__':
    main()