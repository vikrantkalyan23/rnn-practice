from fastapi import FastAPI

from app.schemas import PredictionRequest, PredictionResponse

from app.services.prediction_service import PredictionService


app = FastAPI(title="Next Word Prediction API", version="1.0.0")


prediction_service = PredictionService()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    generated_text = prediction_service.predict(request.text, request.num_words)

    return PredictionResponse(input_text=request.text, generated_text=generated_text)
