from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app.data import load_tokenizer


class NextWordPredictor:
    def __init__(self, model_path: Path, tokenizer_path: Path, sequence_length: int):
        self.model = load_model(model_path)
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.sequence_length = sequence_length
        self.index_to_word = {
            index: word for word, index in self.tokenizer.word_index.items()
        }

    def _prepare_input(self, text: str):
        token_ids = self.tokenizer.texts_to_sequences([text])[0]

        return pad_sequences(
            [token_ids],
            maxlen=self.sequence_length,
            padding="pre",
            truncating="pre",
        )

    def predict_top_words(self, text: str, top_k: int = 5):
        """Return the most likely words and their probabilities."""
        probabilities = self.model.predict(self._prepare_input(text), verbose=0)[0]
        candidate_ids = np.argsort(probabilities)[::-1]

        results = []
        for token_id in candidate_ids:
            word = self.index_to_word.get(int(token_id))
            if word and word != "<OOV>":
                results.append((word, float(probabilities[token_id])))
            if len(results) == top_k:
                break

        return results

    def predict_next_word(self, text: str):
        predictions = self.predict_top_words(text, top_k=1)
        return predictions[0][0] if predictions else None

    def generate_text(self, seed_text: str, next_words: int = 5):
        result = seed_text.strip()

        for _ in range(next_words):
            next_word = self.predict_next_word(result)
            if not next_word:
                break
            result = f"{result} {next_word}"

        return result
