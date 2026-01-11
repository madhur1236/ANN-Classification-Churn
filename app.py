import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

##Load the trained model
# model = tf.keras.models.load_model('model.h5')

#Load trained model for Streamlit cloud
model = tf.keras.models.load_model("model_tf_cloud", compile=False)

##Load the encoders and scalers
with open('label_encode_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('ohe_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


##Streamlit app
st.title("Customer Churn Prediction")

##User input
#with st.form("user_input_form"):
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18,90)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])
    #submit = st.form_submit_button("Predict Churn")

##Prepare the input data
#if submit:
input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumofProduct': [num_of_products],
        'HasCrCredit': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

# --- Predict only when button is clicked ---
if st.button("Predict Churn"):

    ##One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    ##Combine one-hot encoded column with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    ##Scaling the input data
    input_data_scaled = scaler.transform(input_data)

    #Predict
    prediction = model.predict(input_data_scaled)
    prediction_prob = prediction[0][0]

    ##Show Prediction
    if prediction_prob > 0.5:
         st.markdown(f"<p style='color:red; font-size:20px;'>Customer will surely churn ({prediction_prob:.2%})</p>", unsafe_allow_html=True)
    else: 
          st.markdown(f"<p style='color:green; font-size:20px;'>Customer will not churn ({prediction_prob:.2%})</p>", unsafe_allow_html=True)



