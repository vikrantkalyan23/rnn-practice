from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "next_word_model.keras"

TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

SEQUENCE_LENGTH = 5

EMBEDDING_DIM = 128

LSTM_UNITS = 128

DROPOUT = 0.2