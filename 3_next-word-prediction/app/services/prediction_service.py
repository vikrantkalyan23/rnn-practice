from app.model.predictor import NextWordPredictor


class PredictionService:
    def __init__(self):
        self.predictor = NextWordPredictor(
            model_path="models/next_word_model.keras",
            tokenizer_path="models/tokenizer.json",
            sequence_length=20,
        )

    def predict(self, text: str, num_words: int):
        return self.predictor.generate_text(text, num_words)
