import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(BASE_DIR / ".cache" / "matplotlib"),
)

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


# ============================================================
# DATA
# ============================================================

SEQUENCE_LENGTH = 8

# Current corpus:
# 854 unique words
#
# 120 is too aggressive.
MAX_VOCAB_SIZE = 500


# ============================================================
# MODEL
# ============================================================

EMBEDDING_DIM = 96

LSTM_UNITS = 96

DROPOUT = 0.20

L2_STRENGTH = 0.0001


# ============================================================
# OPTIMIZATION
# ============================================================

LEARNING_RATE = 0.001

BATCH_SIZE = 16

EPOCHS = 150


# ============================================================
# DATA SPLIT
# ============================================================

VALIDATION_RATIO = 0.20

TEST_RATIO = 0.15

RANDOM_SEED = 11

TEST_SPLIT_SEED = 2025

VALIDATION_SEEDS = (
    7,
    11,
    19,
)
