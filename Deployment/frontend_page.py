import streamlit as st
import requests
import pandas as pd

# URL of your FastAPI predict endpoint (update if deployed)
PREDICT_URL = "https://loan-api-963580054894.us-central1.run.app/predict"

st.title("Loan Prediction Demo")

with st.form("input_form"):
    person_age = st.slider("Age", min_value=18, max_value=120, value=34)
    person_gender = st.selectbox("Gender", ["female", "male"])
    person_education = st.selectbox("Education", ["Bachelor", "Master", "Doctorate", "Associate", "High School"])
    person_income = st.number_input("Income", value=97265.0)
    person_emp_exp = st.slider("Years employed", min_value=0, max_value=60, value=11)
    person_home_ownership = st.selectbox("Home ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
    loan_amnt = st.slider("Loan amount", min_value=500, max_value=35000, value=15000)
    loan_intent = st.selectbox("Loan intent", 
                               ['PERSONAL', 
                                'EDUCATION', 
                                'MEDICAL', 
                                'VENTURE', 
                                'HOMEIMPROVEMENT',
                                'DEBTCONSOLIDATION']
                            )
    loan_int_rate = st.number_input("Loan interest rate",
                            min_value=5.0,
                            max_value=20.0,
                            value=12.73,
                            step=0.01,
                            format="%.2f"
                            )
    loan_percent_income = st.number_input("Loan percent of income",
                                min_value=0,
                                max_value=100,
                                value=16,
                                step=1,
                                )
    cb_person_cred_hist_length = st.number_input("Credit history length", min_value=0, max_value=102, value=5)
    credit_score = st.number_input("Credit score", min_value=300, max_value=850, value=631)
    previous_loan_defaults_on_file = st.selectbox("Previous defaults", ["No", "Yes"])
    submit = st.form_submit_button("Predict")

if submit:
    instance = {
        "person_age": person_age,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": (loan_percent_income/100.0),
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }

    # send to FastAPI endpoint
    resp = requests.post(PREDICT_URL, json={"instances": [instance]})
    if resp.status_code != 200:
        st.error(f"Request failed: {resp.status_code} {resp.text}")
    else:
        out = resp.json()
        if out.get("predictions")[0] == 1:
            st.success("Prediction: Loan Approved")
        else:
            st.error("Prediction: Loan Denied")