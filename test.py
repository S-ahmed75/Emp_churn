import streamlit as st
from joblib import load
import dill
import pandas as pd
import datetime

# Load the pretrained model
with open('pipeline.pkl', 'rb') as file:
    model = dill.load(file)

my_feature_dict = load('my_feature_dict.pkl')

# Function to predict churn
def predict_churn(data):
    prediction = model.predict(data)
    return prediction



st.title('Employee Churn Prediction App')
st.subheader('Based on Employee Dataset')

col1, col2 = st.columns(2)

# Display categorical features
with col1:
    st.subheader('Categorical Features')
    categorical_input = my_feature_dict.get('CATEGORICAL')
    categorical_input_vals = {}
    for i, col in enumerate(categorical_input.get('Column Name').values()):
        categorical_input_vals[col] = st.selectbox(col, categorical_input.get('Members')[i], key=col)

# Dsiplay numerical features

with col2:
    st.subheader('Numerical Features')
    numerical_input = my_feature_dict.get('NUMERICAL')
    numerical_input_vals = {}

    current_year = datetime.datetime.now().year

    for col in numerical_input.get('Column Name'):
        if col == 'JoiningYear':
            numerical_input_vals[col] = st.selectbox(
                col, list(range(2000, current_year + 1)), key=col
            )
        elif col == 'PaymentTier':
            numerical_input_vals[col] = st.selectbox(
                col, [1, 2, 3], key=col
            )
        else:
            numerical_input_vals[col] = st.number_input(
                col, key=col, step=1, format="%d"
            )

# Combine numerical and categorical input dicts
input_data = dict(list(categorical_input_vals.items()) + list((numerical_input_vals.items())))

input_data= pd.DataFrame.from_dict(input_data,orient='index').T

# Churn Prediction
if st.button('Predict'):
    prediction = predict_churn(input_data)[0]
    translation_dict = {"Leave": "expected", "Stay": "Not Expected"}
    prediction_translate = translation_dict.get(prediction)
    st.write(f'The Prediction is **{prediction}**, Hence employee is **{prediction_translate}** to churn.')
    
    if prediction == "Leave":
        st.image("leave.jpeg", caption="Employee Likely to Churn", width=300)
    else:
        st.image("stay.jpeg", caption="Employee Likely to Stay", width=300)
    
st.subheader('Created by Sarfaraz Ahmed')