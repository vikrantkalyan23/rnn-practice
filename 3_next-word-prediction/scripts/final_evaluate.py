import json
import os
import sys
from pathlib import Path


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PROJECT_DIR / ".cache" / "matplotlib"),
)


# ============================================================
# THIRD-PARTY IMPORTS
# ============================================================

import numpy as np

from tensorflow.keras.models import load_model


# ============================================================
# APPLICATION IMPORTS
# ============================================================

from app.config import (
    DATA_PATH,
    ERROR_ANALYSIS_PATH,
    FINAL_TEST_METRICS_PATH,
    MODEL_CONFIG_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    TEST_RATIO,
    TEST_SPLIT_SEED,
    TOKENIZER_PATH,
    VALIDATION_RATIO,
)

from app.data import (
    clean_text,
    count_possible_targets,
    create_sequences,
    load_tokenizer,
    split_text_train_validation_test,
)


# ============================================================
# CONSTANTS
# ============================================================

TOP_K_VALUES = (1, 3, 5)

EPSILON = 1e-12


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_perplexity(loss: float) -> float:
    """
    Convert cross-entropy loss to perplexity.
    """

    return float(
        np.exp(
            np.clip(
                loss,
                0.0,
                50.0,
            )
        )
    )


def calculate_top_k_accuracy(
    probabilities: np.ndarray,
    targets: np.ndarray,
    k: int,
    ignore_oov: bool = True,
    oov_id: int | None = None,
):
    """
    Calculate Top-K accuracy.

    OOV targets are excluded by default.
    """

    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must have shape "
            "(samples, vocabulary_size)"
        )

    if targets.ndim != 1:
        raise ValueError(
            "targets must have shape (samples,)"
        )

    if len(probabilities) != len(targets):
        raise ValueError(
            "Prediction and target lengths differ."
        )

    valid_mask = np.ones(
        len(targets),
        dtype=bool,
    )

    if (
        ignore_oov
        and oov_id is not None
    ):
        valid_mask &= (
            targets != oov_id
        )

    valid_probabilities = (
        probabilities[valid_mask]
    )

    valid_targets = (
        targets[valid_mask]
    )

    if len(valid_targets) == 0:
        return 0.0

    k = min(
        k,
        probabilities.shape[1],
    )

    top_ids = np.argpartition(
        valid_probabilities,
        -k,
        axis=1,
    )[:, -k:]

    correct = np.any(
        top_ids
        == valid_targets[:, None],
        axis=1,
    )

    return float(
        np.mean(correct)
    )


def calculate_oov_statistics(
    targets: np.ndarray,
    oov_id: int | None,
):
    """
    Calculate OOV statistics for the test set.
    """

    total = len(targets)

    if total == 0:
        return {
            "oov_target_count": 0,
            "oov_target_ratio": 0.0,
            "known_target_count": 0,
            "known_target_ratio": 0.0,
        }

    if oov_id is None:
        return {
            "oov_target_count": 0,
            "oov_target_ratio": 0.0,
            "known_target_count": int(total),
            "known_target_ratio": 1.0,
        }

    oov_mask = (
        targets == oov_id
    )

    oov_count = int(
        np.sum(oov_mask)
    )

    known_count = (
        total - oov_count
    )

    return {
        "oov_target_count": oov_count,
        "oov_target_ratio": float(
            oov_count / total
        ),
        "known_target_count": known_count,
        "known_target_ratio": float(
            known_count / total
        ),
    }


def ranked_predictions(
    probabilities: np.ndarray,
    index_to_word: dict,
    top_k: int = 5,
    oov_id: int | None = None,
):
    """
    Return ranked human-readable predictions.

    <OOV> is intentionally excluded from the displayed
    predictions because it is not a real predicted word.
    """

    ranked_ids = np.argsort(
        probabilities
    )[::-1]

    predictions = []

    for token_id in ranked_ids:

        token_id = int(token_id)

        if (
            oov_id is not None
            and token_id == oov_id
        ):
            continue

        word = index_to_word.get(
            token_id
        )

        if not word:
            continue

        if word in {
            "<OOV>",
            "<PAD>",
        }:
            continue

        predictions.append(
            {
                "word": word,
                "probability": float(
                    probabilities[token_id]
                ),
            }
        )

        if len(predictions) >= top_k:
            break

    return predictions


def get_context_words(
    input_ids: np.ndarray,
    index_to_word: dict,
):
    """
    Convert padded input IDs into readable context.
    """

    words = []

    for token_id in input_ids:

        token_id = int(token_id)

        if token_id == 0:
            continue

        word = index_to_word.get(
            token_id,
            "<UNK>",
        )

        if word == "<OOV>":
            word = "<OOV>"

        words.append(word)

    return words


def calculate_rank(
    probabilities: np.ndarray,
    target_id: int,
):
    """
    Calculate the rank of the actual target.

    Rank 1 means the target had the highest probability.
    """

    target_probability = float(
        probabilities[target_id]
    )

    greater_count = int(
        np.sum(
            probabilities
            > target_probability
            + EPSILON
        )
    )

    return greater_count + 1


# ============================================================
# MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    FINAL_TEST_METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # VERIFY ARTIFACTS
    # --------------------------------------------------------

    required_files = [
        MODEL_PATH,
        TOKENIZER_PATH,
        MODEL_CONFIG_PATH,
        DATA_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print(
        "Loading final model..."
    )

    model = load_model(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # LOAD TOKENIZER
    # --------------------------------------------------------

    tokenizer = load_tokenizer(
        TOKENIZER_PATH
    )

    # --------------------------------------------------------
    # LOAD MODEL CONFIG
    # --------------------------------------------------------

    model_config = json.loads(
        MODEL_CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    # --------------------------------------------------------
    # TOKEN ID -> WORD
    # --------------------------------------------------------

    index_to_word = {
        int(index): word
        for word, index
        in tokenizer.word_index.items()
    }

    # --------------------------------------------------------
    # OOV ID
    # --------------------------------------------------------

    oov_id = tokenizer.word_index.get(
        "<OOV>"
    )

    # --------------------------------------------------------
    # LOAD COMPLETE CORPUS
    # --------------------------------------------------------

    text = DATA_PATH.read_text(
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # RECREATE EXACT FIXED TEST SPLIT
    #
    # IMPORTANT:
    #
    # This must match train.py.
    # --------------------------------------------------------

    (
        train_text,
        validation_text,
        test_text,
    ) = split_text_train_validation_test(
        text,
        validation_ratio=VALIDATION_RATIO,
        test_ratio=TEST_RATIO,
        test_seed=TEST_SPLIT_SEED,
        validation_seed=RANDOM_SEED,
    )

    # --------------------------------------------------------
    # CREATE TEST SEQUENCES
    # --------------------------------------------------------

    sequence_length = int(
        model_config["sequence_length"]
    )

    X_test, y_test = create_sequences(
        test_text,
        tokenizer,
        sequence_length=sequence_length,
    )

    if len(y_test) == 0:
        raise ValueError(
            "No test examples were created."
        )

    print(
        f"Test examples: {len(y_test)}"
    )

    # --------------------------------------------------------
    # TEST LOSS + ACCURACY
    # --------------------------------------------------------

    test_loss, test_accuracy = (
        model.evaluate(
            X_test,
            y_test,
            verbose=0,
        )
    )

    test_loss = float(
        test_loss
    )

    test_accuracy = float(
        test_accuracy
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    probabilities = model.predict(
        X_test,
        verbose=0,
    )

    predicted_ids = np.argmax(
        probabilities,
        axis=1,
    )

    # --------------------------------------------------------
    # MODEL OUTPUT SIZE
    # --------------------------------------------------------

    output_vocab_size = (
        probabilities.shape[1]
    )

    # --------------------------------------------------------
    # OOV STATISTICS
    # --------------------------------------------------------

    oov_stats = calculate_oov_statistics(
        y_test,
        oov_id,
    )

    # --------------------------------------------------------
    # TOP-K METRICS
    # --------------------------------------------------------

    top_k_scores = {}

    for k in TOP_K_VALUES:
        top_k_scores[
            f"top_{k}_accuracy"
        ] = calculate_top_k_accuracy(
            probabilities,
            y_test,
            k,
            ignore_oov=True,
            oov_id=oov_id,
        )

    # --------------------------------------------------------
    # TEST TARGET COVERAGE
    # --------------------------------------------------------

    possible_test_targets = (
        count_possible_targets(
            test_text
        )
    )

    if possible_test_targets > 0:
        test_target_coverage = (
            len(y_test)
            / possible_test_targets
        )
    else:
        test_target_coverage = 0.0

    test_target_coverage = float(
        min(
            1.0,
            test_target_coverage,
        )
    )

    # --------------------------------------------------------
    # ERROR MASK
    # --------------------------------------------------------

    known_target_mask = np.ones(
        len(y_test),
        dtype=bool,
    )

    if oov_id is not None:
        known_target_mask &= (
            y_test != oov_id
        )

    # Only evaluate word prediction errors
    # for known target words.
    known_indices = np.where(
        known_target_mask
    )[0]

    known_targets = y_test[
        known_target_mask
    ]

    known_predictions = predicted_ids[
        known_target_mask
    ]

    known_probabilities = probabilities[
        known_target_mask
    ]

    # --------------------------------------------------------
    # TOP-1 KNOWN-WORD ACCURACY
    # --------------------------------------------------------

    if len(known_targets) > 0:

        known_top1_accuracy = float(
            np.mean(
                known_predictions
                == known_targets
            )
        )

    else:

        known_top1_accuracy = 0.0

    # --------------------------------------------------------
    # TEST ERRORS
    # --------------------------------------------------------

    errors = []

    for local_index, original_index in enumerate(
        known_indices
    ):

        target_id = int(
            y_test[original_index]
        )

        predicted_id = int(
            predicted_ids[original_index]
        )

        # Correct prediction.
        if target_id == predicted_id:
            continue

        row_probabilities = (
            probabilities[
                original_index
            ]
        )

        actual_probability = float(
            row_probabilities[
                target_id
            ]
        )

        predicted_probability = float(
            row_probabilities[
                predicted_id
            ]
        )

        target_rank = calculate_rank(
            row_probabilities,
            target_id,
        )

        context_words = (
            get_context_words(
                X_test[
                    original_index
                ],
                index_to_word,
            )
        )

        actual_word = index_to_word.get(
            target_id,
            "<UNK>",
        )

        predicted_word = index_to_word.get(
            predicted_id,
            "<UNK>",
        )

        errors.append(
            {
                "test_example": int(
                    original_index + 1
                ),

                "context": " ".join(
                    context_words
                ),

                "actual": actual_word,

                "predicted": predicted_word,

                "target_rank": int(
                    target_rank
                ),

                "actual_probability": (
                    actual_probability
                ),

                "predicted_probability": (
                    predicted_probability
                ),

                "probability_margin": (
                    predicted_probability
                    - actual_probability
                ),

                "top_predictions": (
                    ranked_predictions(
                        row_probabilities,
                        index_to_word,
                        top_k=5,
                        oov_id=oov_id,
                    )
                ),
            }
        )

    # --------------------------------------------------------
    # SORT ERRORS
    # --------------------------------------------------------

    # Lowest probability assigned to the actual word first.
    errors.sort(
        key=lambda item:
        item["actual_probability"]
    )

    # --------------------------------------------------------
    # RANK DISTRIBUTION
    # --------------------------------------------------------

    ranks = []

    for original_index in known_indices:

        target_id = int(
            y_test[
                original_index
            ]
        )

        rank = calculate_rank(
            probabilities[
                original_index
            ],
            target_id,
        )

        ranks.append(rank)

    ranks = np.asarray(
        ranks,
        dtype=np.int32,
    )

    if len(ranks) > 0:

        mean_target_rank = float(
            np.mean(ranks)
        )

        median_target_rank = float(
            np.median(ranks)
        )

        top1_rank_ratio = float(
            np.mean(ranks <= 1)
        )

        top3_rank_ratio = float(
            np.mean(ranks <= 3)
        )

        top5_rank_ratio = float(
            np.mean(ranks <= 5)
        )

    else:

        mean_target_rank = 0.0
        median_target_rank = 0.0
        top1_rank_ratio = 0.0
        top3_rank_ratio = 0.0
        top5_rank_ratio = 0.0

    # --------------------------------------------------------
    # TEST ERROR RATE
    # --------------------------------------------------------

    if len(known_targets) > 0:

        known_error_count = int(
            np.sum(
                known_predictions
                != known_targets
            )
        )

        known_error_rate = float(
            known_error_count
            / len(known_targets)
        )

    else:

        known_error_count = 0
        known_error_rate = 0.0

    # --------------------------------------------------------
    # CONFIDENCE STATISTICS
    # --------------------------------------------------------

    if len(known_indices) > 0:

        actual_probabilities = (
            probabilities[
                known_indices,
                y_test[
                    known_indices
                ],
            ]
        )

        mean_actual_probability = float(
            np.mean(
                actual_probabilities
            )
        )

        median_actual_probability = float(
            np.median(
                actual_probabilities
            )
        )

        max_actual_probability = float(
            np.max(
                actual_probabilities
            )
        )

        min_actual_probability = float(
            np.min(
                actual_probabilities
            )
        )

    else:

        mean_actual_probability = 0.0
        median_actual_probability = 0.0
        max_actual_probability = 0.0
        min_actual_probability = 0.0

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = {
        "evaluation_type": (
            "final_untouched_test"
        ),

        "evaluation_protocol": (
            "fixed-test-multi-validation-v1"
        ),

        "split": "untouched_test",

        "test_split_seed": int(
            TEST_SPLIT_SEED
        ),

        "validation_seed": int(
            RANDOM_SEED
        ),

        "validation_ratio": float(
            VALIDATION_RATIO
        ),

        "test_ratio": float(
            TEST_RATIO
        ),

        "sequence_length": sequence_length,

        "model_vocab_size": int(
            output_vocab_size
        ),

        "max_vocab_size": model_config.get(
            "max_vocab_size"
        ),

        "embedding_dim": model_config.get(
            "embedding_dim"
        ),

        "lstm_units": model_config.get(
            "lstm_units"
        ),

        "dropout": model_config.get(
            "dropout"
        ),

        "l2_strength": model_config.get(
            "l2_strength"
        ),

        "learning_rate": model_config.get(
            "learning_rate"
        ),

        "batch_size": model_config.get(
            "batch_size"
        ),

        "train_lines": len(
            train_text.splitlines()
        ),

        "validation_lines": len(
            validation_text.splitlines()
        ),

        "test_lines": len(
            test_text.splitlines()
        ),

        "test_examples": int(
            len(y_test)
        ),

        "test_target_count_possible": int(
            possible_test_targets
        ),

        "test_target_count_evaluated": int(
            len(y_test)
        ),

        "test_target_coverage": (
            test_target_coverage
        ),

        "test_loss": test_loss,

        "test_perplexity": (
            calculate_perplexity(
                test_loss
            )
        ),

        "test_accuracy": test_accuracy,

        "known_target_top_1_accuracy": (
            known_top1_accuracy
        ),

        "known_target_error_rate": (
            known_error_rate
        ),

        "known_target_error_count": (
            known_error_count
        ),

        "top_1_accuracy": top_k_scores[
            "top_1_accuracy"
        ],

        "top_3_accuracy": top_k_scores[
            "top_3_accuracy"
        ],

        "top_5_accuracy": top_k_scores[
            "top_5_accuracy"
        ],

        "mean_target_rank": (
            mean_target_rank
        ),

        "median_target_rank": (
            median_target_rank
        ),

        "target_rank_top_1": (
            top1_rank_ratio
        ),

        "target_rank_top_3": (
            top3_rank_ratio
        ),

        "target_rank_top_5": (
            top5_rank_ratio
        ),

        "mean_actual_word_probability": (
            mean_actual_probability
        ),

        "median_actual_word_probability": (
            median_actual_probability
        ),

        "maximum_actual_word_probability": (
            max_actual_probability
        ),

        "minimum_actual_word_probability": (
            min_actual_probability
        ),

        **oov_stats,
    }

    # --------------------------------------------------------
    # ERROR ANALYSIS
    # --------------------------------------------------------

    analysis = {
        "evaluation_type": (
            "final_untouched_test_error_analysis"
        ),

        "test_examples": int(
            len(y_test)
        ),

        "known_target_examples": int(
            len(known_targets)
        ),

        "total_errors": int(
            len(errors)
        ),

        "error_rate": (
            known_error_rate
        ),

        "oov_target_count": (
            oov_stats[
                "oov_target_count"
            ]
        ),

        "oov_target_ratio": (
            oov_stats[
                "oov_target_ratio"
            ]
        ),

        "mean_target_rank": (
            mean_target_rank
        ),

        "median_target_rank": (
            median_target_rank
        ),

        "lowest_confidence_actual_word_errors": (
            errors[:20]
        ),

        "largest_probability_margin_errors": sorted(
            errors,
            key=lambda item:
            item["probability_margin"],
            reverse=True,
        )[:20],

        "worst_ranked_errors": sorted(
            errors,
            key=lambda item:
            item["target_rank"],
            reverse=True,
        )[:20],
    }

    # --------------------------------------------------------
    # SAVE FINAL METRICS
    # --------------------------------------------------------

    FINAL_TEST_METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # SAVE ERROR ANALYSIS
    # --------------------------------------------------------

    ERROR_ANALYSIS_PATH.write_text(
        json.dumps(
            analysis,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL UNTOUCHED TEST EVALUATION")
    print("=" * 60)

    print()
    print("Evaluation protocol")
    print("-" * 35)

    print(
        f"Test seed        : "
        f"{TEST_SPLIT_SEED}"
    )

    print(
        f"Test lines       : "
        f"{metrics['test_lines']}"
    )

    print(
        f"Test examples    : "
        f"{metrics['test_examples']}"
    )

    print()
    print("Loss")
    print("-" * 35)

    print(
        f"Test loss        : "
        f"{metrics['test_loss']:.4f}"
    )

    print(
        f"Perplexity       : "
        f"{metrics['test_perplexity']:.2f}"
    )

    print()
    print("Accuracy")
    print("-" * 35)

    print(
        f"Top-1            : "
        f"{metrics['top_1_accuracy']:.2%}"
    )

    print(
        f"Top-3            : "
        f"{metrics['top_3_accuracy']:.2%}"
    )

    print(
        f"Top-5            : "
        f"{metrics['top_5_accuracy']:.2%}"
    )

    print(
        f"Known Top-1      : "
        f"{metrics['known_target_top_1_accuracy']:.2%}"
    )

    print()
    print("Target coverage")
    print("-" * 35)

    print(
        f"Coverage         : "
        f"{metrics['test_target_coverage']:.2%}"
    )

    print(
        f"OOV targets      : "
        f"{metrics['oov_target_ratio']:.2%}"
    )

    print(
        f"Known targets    : "
        f"{metrics['known_target_ratio']:.2%}"
    )

    print()
    print("Ranking")
    print("-" * 35)

    print(
        f"Mean target rank : "
        f"{metrics['mean_target_rank']:.2f}"
    )

    print(
        f"Median rank      : "
        f"{metrics['median_target_rank']:.2f}"
    )

    print(
        f"Rank <= 1        : "
        f"{metrics['target_rank_top_1']:.2%}"
    )

    print(
        f"Rank <= 3        : "
        f"{metrics['target_rank_top_3']:.2%}"
    )

    print(
        f"Rank <= 5        : "
        f"{metrics['target_rank_top_5']:.2%}"
    )

    print()
    print("Confidence")
    print("-" * 35)

    print(
        f"Mean target prob : "
        f"{metrics['mean_actual_word_probability']:.4f}"
    )

    print(
        f"Median target    : "
        f"{metrics['median_actual_word_probability']:.4f}"
    )

    print()
    print("Errors")
    print("-" * 35)

    print(
        f"Known errors     : "
        f"{metrics['known_target_error_count']}"
    )

    print(
        f"Known error rate : "
        f"{metrics['known_target_error_rate']:.2%}"
    )

    print()
    print(
        f"Metrics saved to : "
        f"{FINAL_TEST_METRICS_PATH}"
    )

    print(
        f"Errors saved to  : "
        f"{ERROR_ANALYSIS_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()