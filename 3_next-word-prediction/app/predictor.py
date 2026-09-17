from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app.data import clean_text, load_tokenizer


END_TOKEN = "<END>"


class CorpusBackoff:
    """Look up next words using the longest context observed in training text."""

    def __init__(self, text: str, max_context: int):
        self.max_context = max_context
        self.counts = defaultdict(Counter)

        for line in clean_text(text).splitlines():
            words = line.split()
            targets = words[1:] + [END_TOKEN]
            for target_index, target in enumerate(targets, start=1):
                for length in range(1, min(max_context, target_index) + 1):
                    context = tuple(words[target_index - length : target_index])
                    self.counts[context][target] += 1

    def probabilities(self, text: str, temperature: float = 1.0):
        words = clean_text(text).split()
        for length in range(min(len(words), self.max_context), 0, -1):
            candidates = self.counts.get(tuple(words[-length:]))
            if candidates:
                words_and_counts = list(candidates.items())
                weights = np.asarray(
                    [count for _, count in words_and_counts], dtype=np.float64
                )
                weights = np.power(weights, 1.0 / temperature)
                weights /= weights.sum()
                return {
                    word: float(weight)
                    for (word, _), weight in zip(words_and_counts, weights)
                }
        return {}


class NextWordPredictor:
    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Path,
        sequence_length: int,
        training_text_path: Path | None = None,
    ):
        self.model = load_model(model_path)
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.sequence_length = sequence_length
        self.index_to_word = {
            index: word for word, index in self.tokenizer.word_index.items()
        }
        self.corpus_backoff = None
        if training_text_path and training_text_path.exists():
            self.corpus_backoff = CorpusBackoff(
                training_text_path.read_text(encoding="utf-8"),
                sequence_length,
            )

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
        return [
            candidate
            for candidate in self._ranked_candidates(text, temperature)
            if candidate[0] != END_TOKEN
        ][:top_k]

    def _ranked_candidates(self, text: str, temperature: float):
        neural = self._predict_probabilities(text, temperature)
        combined = {
            word: 0.15 * float(neural[token_id])
            for token_id, word in self.index_to_word.items()
            if word != "<OOV>" and token_id < len(neural)
        }

        corpus = (
            self.corpus_backoff.probabilities(text, temperature)
            if self.corpus_backoff
            else {}
        )
        if corpus:
            for word, probability in corpus.items():
                combined[word] = combined.get(word, 0.0) + 0.85 * probability
        else:
            combined = {
                word: probability / 0.15 for word, probability in combined.items()
            }

        total = sum(combined.values())
        return sorted(
            ((word, probability / total) for word, probability in combined.items()),
            key=lambda candidate: candidate[1],
            reverse=True,
        )

    def predict_next_word(self, text: str, temperature: float = 1.0):
        predictions = self._ranked_candidates(text, temperature)
        if not predictions or predictions[0][0] == END_TOKEN:
            return None
        return predictions[0][0]

    def sample_next_word(self, text: str, temperature: float = 1.0, top_k: int = 5):
        """Sample one next word from the top-k distribution."""
        candidates = self._ranked_candidates(text, temperature)[:top_k]
        if not candidates:
            return None

        normalized = np.asarray(
            [probability for _, probability in candidates], dtype=np.float64
        )
        normalized = normalized / normalized.sum()
        sampled_index = int(np.random.choice(len(candidates), p=normalized))
        sampled_word = candidates[sampled_index][0]
        return None if sampled_word == END_TOKEN else sampled_word

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
