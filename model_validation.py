import joblib
import pandas as pd

obj = joblib.load('models/best_model.joblib')
pipe = obj['pipeline']
features = obj['features']

sample_row = {
    "person_age": 22.0,
    "person_gender": "male",
    "person_education": "Master",
    "person_income": 66135.0,
    "person_emp_exp": 1,
    "person_home_ownership": "RENT",
    "loan_amnt": 35000.0,
    "loan_intent": "MEDICAL",
    "loan_int_rate": 14.27,
    "loan_percent_income": 0.53,
    "cb_person_cred_hist_length": 4.0,
    "credit_score": 586,
    "previous_loan_defaults_on_file": "NO"
}

# prepare a DataFrame with the same columns/order as `features`
X_sample = pd.DataFrame([sample_row])

for f in features:
    if f not in X_sample.columns:
        print(f"Adding missing feature column: {f}")
        raise ValueError(f"Missing feature: {f}")
pred = pipe.predict(X_sample[features])
print('predicted label indices:', pred)