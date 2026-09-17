import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import MODEL_CONFIG_PATH, MODEL_PATH, TOKENIZER_PATH, TRAINING_TEXT_PATH
from app.predictor import NextWordPredictor
from app.schemas import PredictionRequest, PredictionResponse


app = FastAPI(title="Next Word Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model_config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
predictor = NextWordPredictor(
    model_path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    sequence_length=model_config["sequence_length"],
    training_text_path=TRAINING_TEXT_PATH,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    top_predictions = predictor.predict_top_words(
        request.text,
        top_k=request.top_k,
        temperature=request.temperature,
    )
    generated_text = predictor.generate_text(
        request.text,
        next_words=request.num_words,
        temperature=request.temperature,
        top_k=request.top_k,
        sample=request.sample,
    )

    return PredictionResponse(
        input_text=request.text,
        generated_text=generated_text,
        predictions=[
            {"word": word, "probability": probability}
            for word, probability in top_predictions
        ],
    )
