import os
from pathlib import Path


# All paths are absolute, so commands work from any current directory.
BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".cache" / "matplotlib"))
DATA_PATH = BASE_DIR / "data" / "raw" / "corpus.txt"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_corpus.txt"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "next_word_model.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
MODEL_CONFIG_PATH = MODEL_DIR / "config.json"
HISTORY_PATH = MODEL_DIR / "training_history.json"

OUTPUT_DIR = BASE_DIR / "outputs"

# Model and training defaults are intentionally small for this tiny corpus.
SEQUENCE_LENGTH = 5
MAX_VOCAB_SIZE = 80
EMBEDDING_DIM = 24
LSTM_UNITS = 24
DROPOUT = 0.35
LEARNING_RATE = 0.001
VALIDATION_RATIO = 0.2
BATCH_SIZE = 8
EPOCHS = 150
RANDOM_SEED = 11
