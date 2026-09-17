import json

from fastapi import FastAPI

from app.config import MODEL_CONFIG_PATH, MODEL_PATH, TOKENIZER_PATH
from app.predictor import NextWordPredictor
from app.schemas import PredictionRequest, PredictionResponse


app = FastAPI(title="Next Word Prediction API", version="1.0.0")


model_config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
predictor = NextWordPredictor(
    model_path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    sequence_length=model_config["sequence_length"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    generated_text = predictor.generate_text(request.text, request.num_words)

    return PredictionResponse(input_text=request.text, generated_text=generated_text)
