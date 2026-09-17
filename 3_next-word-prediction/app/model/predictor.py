import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app.preprocessing.tokenizer import load_tokenizer


class NextWordPredictor:
    def __init__(
        self,
        model_path,
        tokenizer_path,
        sequence_length,
    ):
        self.model = load_model(model_path)

        self.tokenizer = load_tokenizer(tokenizer_path)

        self.sequence_length = sequence_length

    def predict_next_word(
        self,
        text: str,
    ):
        token_list = self.tokenizer.texts_to_sequences([text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=self.sequence_length,
            padding="pre",
        )

        predictions = self.model.predict(
            token_list,
            verbose=0,
        )

        predicted_id = int(np.argmax(predictions[0]))

        index_word = {index: word for word, index in self.tokenizer.word_index.items()}

        return index_word.get(predicted_id)

    def generate_text(
        self,
        seed_text: str,
        next_words: int = 5,
    ):
        result = seed_text

        for _ in range(next_words):
            next_word = self.predict_next_word(result)

            if not next_word:
                break

            result += " " + next_word

        return result
