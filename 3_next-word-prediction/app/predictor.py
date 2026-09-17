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

    def _predict_probabilities(self, text: str, temperature: float = 1.0):
        probabilities = self.model.predict(self._prepare_input(text), verbose=0)[0]

        if temperature <= 0:
            raise ValueError("Temperature must be greater than zero.")
        if temperature == 1.0:
            return probabilities

        adjusted = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
        adjusted = np.exp(adjusted - np.max(adjusted))
        return adjusted / np.sum(adjusted)

    def predict_top_words(self, text: str, top_k: int = 5, temperature: float = 1.0):
        """Return the most likely words and their probabilities."""
        probabilities = self._predict_probabilities(text, temperature)
        candidate_ids = np.argsort(probabilities)[::-1]

        results = []
        for token_id in candidate_ids:
            word = self.index_to_word.get(int(token_id))
            if word and word != "<OOV>":
                results.append((word, float(probabilities[token_id])))
            if len(results) == top_k:
                break

        return results

    def predict_next_word(self, text: str, temperature: float = 1.0):
        predictions = self.predict_top_words(text, top_k=1, temperature=temperature)
        return predictions[0][0] if predictions else None

    def sample_next_word(self, text: str, temperature: float = 1.0, top_k: int = 5):
        """Sample one next word from the top-k distribution."""
        probabilities = self._predict_probabilities(text, temperature)
        candidate_ids = []
        candidate_probabilities = []

        for token_id in np.argsort(probabilities)[::-1]:
            word = self.index_to_word.get(int(token_id))
            if word and word != "<OOV>":
                candidate_ids.append(int(token_id))
                candidate_probabilities.append(float(probabilities[token_id]))
            if len(candidate_ids) == top_k:
                break

        if not candidate_ids:
            return None

        normalized = np.asarray(candidate_probabilities, dtype=np.float64)
        normalized = normalized / normalized.sum()
        sampled_id = np.random.choice(candidate_ids, p=normalized)
        return self.index_to_word.get(int(sampled_id))

    def generate_text(
        self,
        seed_text: str,
        next_words: int = 5,
        temperature: float = 1.0,
        top_k: int = 5,
        sample: bool = False,
    ):
        result = seed_text.strip()

        for _ in range(next_words):
            if sample:
                next_word = self.sample_next_word(result, temperature, top_k)
            else:
                next_word = self.predict_next_word(result, temperature)
            if not next_word:
                break
            result = f"{result} {next_word}"

        return result
