from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.predictor import TemperaturePredictor


app = FastAPI(title="Temperature RNN API", version="1.0.0")


# Load model once when application starts
predictor = TemperaturePredictor()


class PredictionRequest(BaseModel):
    temperatures: list[float]


@app.get("/")
def root():
    return {"message": "Temperature RNN API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        prediction = predictor.predict(request.temperatures)

        return {"predicted_temperature": prediction}

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
