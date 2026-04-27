import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn.")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.header("Enter Customer Details")

age = st.number_input("Age", min_value=18, max_value=100, value=30)

frequent_flyer = st.selectbox("Frequent Flyer", ["No", "Yes"])
annual_income = st.selectbox("Annual Income Class", ["Low", "Medium", "High"])
services_opted = st.number_input("Services Opted", min_value=1, max_value=10, value=3)
social_media = st.selectbox("Account Synced To Social Media", ["No", "Yes"])
booked_hotel = st.selectbox("Booked Hotel Or Not", ["No", "Yes"])

input_df = pd.DataFrame([{
    "Age": age,
    "FrequentFlyer_Encoded": 1 if frequent_flyer == "Yes" else 0,
    "AnnualIncomeClass_Encoded": {"Low": 0, "Medium": 1, "High": 2}[annual_income],
    "ServicesOpted": services_opted,
    "AccountSyncedToSocialMedia_Encoded": 1 if social_media == "Yes" else 0,
    "BookedHotelOrNot_Encoded": 1 if booked_hotel == "Yes" else 0
}])

if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"Customer is likely to churn. Probability: {probability:.2f}")
        else:
            st.success(f"Customer is not likely to churn. Probability: {probability:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
