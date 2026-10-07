from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["model"] = joblib.load('credit_risk_model.pkl')  # Load the model once when the API starts
    ml_model['threshold'] = joblib.load('best_threshold.pkl')  # Load the threshold once when the API starts
    yield
    ml_model.clear()


app = FastAPI(lifespan=lifespan)


# The only columns that user will see and provide inputs.
class LoanApplication(BaseModel):  # Pydantic Model (Validation)
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


@app.post('/predict')
def predict(data: LoanApplication):
    input_df = pd.DataFrame([data.dict()])

    probability = ml_model['model'].predict_proba(input_df)[:, 1][0]

    prediction = int(probability >= ml_model["threshold"])

    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": ml_model["threshold"],
        "Result": "High Risk" if prediction == 1 else "Low Risk"
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
