import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()


class Date(BaseModel):
    year: int
    month: int


class Prediction(BaseModel):
    prediction: float


@app.post("/predict")
async def predict(payload: Date):
    full_row = [payload.year, payload.month, 1, 0, 0, 0, 1, 0]
    with open('final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict([full_row])[0]
    response = Prediction(prediction=prediction)
    return response
