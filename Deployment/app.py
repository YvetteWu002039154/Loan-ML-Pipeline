from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

class Instance(BaseModel):
    person_age: float
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: float
    person_home_ownership: str
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    credit_score: float
    previous_loan_defaults_on_file: str

class Payload(BaseModel):
    instances: list[Instance]

app = FastAPI()

pipe = joblib.load('models/best_model.joblib')

@app.post("/predict")
def predict(payload: Payload):
    # convert payload.instances to DataFrame or numpy array
    df = pd.DataFrame([inst.dict() for inst in payload.instances])
    # if your pipeline accepts arrays:
    arr = df.values  # or keep DataFrame if you configured it accordingly
    preds = pipe.predict(arr)
    return {"predictions": preds.tolist()}