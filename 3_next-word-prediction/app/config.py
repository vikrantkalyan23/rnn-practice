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
TRAINING_TEXT_PATH = MODEL_DIR / "training_text.txt"

OUTPUT_DIR = BASE_DIR / "outputs"
FINAL_TEST_METRICS_PATH = OUTPUT_DIR / "final_test_metrics.json"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "error_analysis.json"
LR_REFINEMENT_PATH = OUTPUT_DIR / "learning_rate_refinement.json"
MULTI_SPLIT_METRICS_PATH = OUTPUT_DIR / "multi_split_metrics.json"

# Model and training defaults sized for the expanded educational corpus.
SEQUENCE_LENGTH = 7
MAX_VOCAB_SIZE = 120
EMBEDDING_DIM = 48
LSTM_UNITS = 48
DROPOUT = 0.30
L2_STRENGTH = 0.0005
LEARNING_RATE = 0.0015
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.15
BATCH_SIZE = 16
EPOCHS = 150
RANDOM_SEED = 11
TEST_SPLIT_SEED = 2025
VALIDATION_SEEDS = (7, 11, 19)
