import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

# Load the trained model
model=tf.keras.models.load_model('model.keras')

# Load thge encoders and scaler

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

with open('geo_oh_encoder.pkl', 'rb') as file:
    geo_oh_encoder= pickle.load(file)

with open('label_encoder_Gender.pkl','rb') as file:
    label_encoder_Gender= pickle.load(file)
    

# Streamlit app
st.title("Customer Churn Prediction")

# User Input 

geography = st.selectbox("Geography", geo_oh_encoder.categories_[0])
gender= st.selectbox("Gender", label_encoder_Gender.classes_)
age= st.slider("Age", min_value=18, max_value=100, value=30)
balance= st.number_input("Balance")
credit_score= st.number_input("Credit Score")
estimated_salary= st.number_input("Estimated Salary")
tenure= st.slider("Tenure", min_value=0, max_value=10, value=3)
num_of_products= st.slider("Number of Products",1,4)
has_cr_card= st.selectbox("Has Credit Card", [0,1])
is_active_member= st.selectbox("Is Active Member", [0,1])

# Prepare the input data

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_Gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode the Geography feature

geo_encoded = geo_oh_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=geo_oh_encoder.get_feature_names_out(['Geography'])
)
    

# Concatenate the one-hot encoded Geography features with the input data

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data

input_data_scaled = scaler.transform(input_data)

#Make prediction

prediction= model.predict(input_data_scaled)
prediction_probability= prediction[0][0]

st.write(f"Prediction Probability of Churn: {prediction_probability:.2f}")

if prediction_probability > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")
