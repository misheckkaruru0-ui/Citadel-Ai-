from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load your trained ML model
model = joblib.load("citadel_gold_elite_xgb.pkl")

class InputData(BaseModel):
    features: list

@app.get("/")
def read_root():
    return {"message": "Citadel-AI running"}

@app.post("/predict")
def predict(data: InputData):
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
