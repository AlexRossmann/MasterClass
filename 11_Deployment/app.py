
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import statsmodels.api as sm

app = FastAPI()
model = joblib.load("results.pkl")

class Features(BaseModel):
    TESTRESULT: float

@app.post("/predict")
def predict(data: Features):
    X = sm.add_constant([[data.TESTRESULT]], has_constant='add')
    prediction = model.predict(X)[0]
    return {
        "prediction": round(float(prediction), 3)
    }
