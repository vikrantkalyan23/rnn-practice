import json
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache" / "matplotlib"))

import numpy as np  # noqa: E402
from tensorflow.keras.models import load_model  # noqa: E402

from app.config import (  # noqa: E402
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
from app.data import (  # noqa: E402
    count_possible_targets,
    create_sequences,
    load_tokenizer,
    split_text_train_validation_test,
)


def calculate_top_k_accuracy(probabilities, targets, k):
    k = min(k, probabilities.shape[1])
    top_ids = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top_ids == targets[:, None], axis=1)))


def ranked_predictions(probabilities, index_to_word, top_k=5):
    predictions = []
    for token_id in np.argsort(probabilities)[::-1]:
        word = index_to_word.get(int(token_id))
        if word and word != "<OOV>":
            predictions.append(
                {"word": word, "probability": float(probabilities[token_id])}
            )
        if len(predictions) == top_k:
            break
    return predictions


def main():
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    model_config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    model = load_model(MODEL_PATH)
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}

    text = DATA_PATH.read_text(encoding="utf-8")
    train_text, validation_text, test_text = split_text_train_validation_test(
        text,
        validation_ratio=VALIDATION_RATIO,
        test_ratio=TEST_RATIO,
        test_seed=TEST_SPLIT_SEED,
        validation_seed=RANDOM_SEED,
    )
    X_test, y_test = create_sequences(
        test_text,
        tokenizer,
        sequence_length=model_config["sequence_length"],
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    probabilities = model.predict(X_test, verbose=0)
    predicted_ids = np.argmax(probabilities, axis=1)
    top_k_scores = {
        f"top_{k}_accuracy": calculate_top_k_accuracy(probabilities, y_test, k)
        for k in (1, 3, 5)
    }
    perplexity = float(np.exp(min(test_loss, 50)))

    metrics = {
        "split": "untouched_test",
        "test_split_seed": TEST_SPLIT_SEED,
        "validation_seed": RANDOM_SEED,
        "evaluation_protocol": "fixed-test-multi-validation-v1",
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": TEST_RATIO,
        "train_lines": len(train_text.splitlines()),
        "validation_lines": len(validation_text.splitlines()),
        "test_lines": len(test_text.splitlines()),
        "test_examples": int(len(y_test)),
        "test_target_coverage": len(y_test) / count_possible_targets(test_text),
        "test_loss": float(test_loss),
        "test_perplexity": perplexity,
        "test_accuracy": float(test_accuracy),
        **top_k_scores,
    }

    errors = []
    for row_index, (target_id, predicted_id) in enumerate(zip(y_test, predicted_ids)):
        if target_id == predicted_id:
            continue
        token_context = [
            index_to_word.get(int(token_id), "<PAD>")
            for token_id in X_test[row_index]
            if token_id != 0
        ]
        errors.append(
            {
                "context": " ".join(token_context),
                "actual": index_to_word.get(int(target_id), "<UNK>"),
                "predicted": index_to_word.get(int(predicted_id), "<UNK>"),
                "actual_probability": float(probabilities[row_index, target_id]),
                "predicted_probability": float(probabilities[row_index, predicted_id]),
                "top_predictions": ranked_predictions(
                    probabilities[row_index],
                    index_to_word,
                    top_k=5,
                ),
            }
        )

    errors.sort(key=lambda item: item["actual_probability"])
    analysis = {
        "total_errors": len(errors),
        "error_rate": len(errors) / len(y_test),
        "lowest_confidence_actual_word_errors": errors[:20],
    }

    FINAL_TEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_TEST_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ERROR_ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    print("Untouched test metrics")
    print("-" * 35)
    print(f"Loss       : {metrics['test_loss']:.4f}")
    print(f"Perplexity : {metrics['test_perplexity']:.2f}")
    print(f"Top-1      : {metrics['top_1_accuracy']:.2%}")
    print(f"Top-3      : {metrics['top_3_accuracy']:.2%}")
    print(f"Top-5      : {metrics['top_5_accuracy']:.2%}")
    print(f"Examples   : {metrics['test_examples']}")
    print(f"Errors     : {analysis['total_errors']}")
    print(f"\nMetrics saved to: {FINAL_TEST_METRICS_PATH}")
    print(f"Errors saved to : {ERROR_ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
