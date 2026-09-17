import json
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorflow.keras.models import load_model  # noqa: E402

from app.config import (  # noqa: E402
    DATA_PATH,
    HISTORY_PATH,
    MODEL_CONFIG_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    TOKENIZER_PATH,
    VALIDATION_RATIO,
)
from app.data import (  # noqa: E402
    count_possible_targets,
    create_sequences,
    load_tokenizer,
    split_text_by_line,
)


def plot_training_history(history, best_epoch):
    epochs = np.arange(1, len(history["loss"]) + 1)
    best_index = best_epoch - 1

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history["loss"], label="Training loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation loss")
    axes[0].scatter(
        best_epoch,
        history["val_loss"][best_index],
        color="red",
        label=f"Best epoch: {best_epoch}",
        zorder=3,
    )
    axes[0].set_title("Training vs Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history["accuracy"], label="Training accuracy")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation accuracy")
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_history.png", dpi=160)
    plt.close(figure)


def calculate_top_k_accuracy(probabilities, targets, k):
    k = min(k, probabilities.shape[1])
    top_ids = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top_ids == targets[:, None], axis=1)))


def plot_top_k_accuracy(scores):
    labels = [f"Top-{k}" for k in scores]
    values = list(scores.values())

    figure, axis = plt.subplots(figsize=(7, 5))
    bars = axis.bar(labels, values, color=["#2878B5", "#F28E2B", "#59A14F"])
    axis.set_title("Validation Top-k Accuracy")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.1%}",
            ha="center",
        )

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "top_k_accuracy.png", dpi=160)
    plt.close(figure)


def plot_baseline_comparison(model_accuracy, baseline_accuracy, random_accuracy):
    labels = ["Random guess", "Most frequent word", "LSTM model"]
    values = [random_accuracy, baseline_accuracy, model_accuracy]

    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, values, color=["#BAB0AC", "#F28E2B", "#2878B5"])
    axis.set_title("Validation Accuracy vs Simple Baselines")
    axis.set_ylabel("Top-1 accuracy")
    axis.set_ylim(0, max(0.2, max(values) * 1.3))
    axis.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.1%}",
            ha="center",
        )

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "baseline_comparison.png", dpi=160)
    plt.close(figure)


def plot_target_confidence(probabilities, targets, predicted_ids):
    example_numbers = np.arange(1, len(targets) + 1)
    true_word_probability = probabilities[example_numbers - 1, targets]
    correct = predicted_ids == targets
    colors = np.where(correct, "#59A14F", "#E15759")

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(example_numbers, true_word_probability, color=colors)
    axis.set_title("Model Confidence in the Correct Validation Word")
    axis.set_xlabel("Validation example")
    axis.set_ylabel("Probability assigned to correct word")
    upper_limit = max(0.02, float(np.max(true_word_probability)) * 1.25)
    axis.set_ylim(0, upper_limit)
    axis.grid(axis="y", alpha=0.25)
    axis.text(
        0.99,
        0.95,
        "Green = top-1 correct\nRed = top-1 incorrect",
        ha="right",
        va="top",
        transform=axis.transAxes,
    )

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "validation_confidence.png", dpi=160)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    model_config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    model = load_model(MODEL_PATH)

    text = DATA_PATH.read_text(encoding="utf-8")
    train_text, validation_text = split_text_by_line(
        text,
        VALIDATION_RATIO,
        saved_history["random_seed"],
    )
    X_train, y_train = create_sequences(
        train_text,
        tokenizer,
        sequence_length=model_config["sequence_length"],
    )
    X_validation, y_validation = create_sequences(
        validation_text,
        tokenizer,
        sequence_length=model_config["sequence_length"],
    )
    total_validation_targets = count_possible_targets(validation_text)
    validation_coverage = len(y_validation) / total_validation_targets

    train_loss, train_accuracy = model.evaluate(
        X_train,
        y_train,
        verbose=0,
    )
    validation_loss, validation_accuracy = model.evaluate(
        X_validation,
        y_validation,
        verbose=0,
    )
    probabilities = model.predict(X_validation, verbose=0)
    predicted_ids = np.argmax(probabilities, axis=1)

    top_k_scores = {
        k: calculate_top_k_accuracy(probabilities, y_validation, k)
        for k in (1, 3, 5)
    }

    target_counts = np.bincount(y_train, minlength=probabilities.shape[1])
    most_frequent_target = int(np.argmax(target_counts))
    baseline_accuracy = float(np.mean(y_validation == most_frequent_target))
    random_accuracy = 1.0 / probabilities.shape[1]

    perplexity = float(np.exp(min(validation_loss, 50)))
    metrics = {
        "validation_loss": float(validation_loss),
        "train_loss": float(train_loss),
        "loss_gap": float(validation_loss - train_loss),
        "validation_perplexity": perplexity,
        "validation_accuracy": float(validation_accuracy),
        "train_accuracy": float(train_accuracy),
        "accuracy_gap": float(train_accuracy - validation_accuracy),
        "top_1_accuracy": top_k_scores[1],
        "top_3_accuracy": top_k_scores[3],
        "top_5_accuracy": top_k_scores[5],
        "most_frequent_word_baseline": baseline_accuracy,
        "random_guess_baseline": random_accuracy,
        "validation_examples": int(len(y_validation)),
        "validation_target_coverage": validation_coverage,
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    plot_training_history(saved_history["history"], saved_history["best_epoch"])
    plot_top_k_accuracy(top_k_scores)
    plot_baseline_comparison(
        top_k_scores[1],
        baseline_accuracy,
        random_accuracy,
    )
    plot_target_confidence(probabilities, y_validation, predicted_ids)

    print("Validation metrics")
    print("-" * 35)
    print(f"Loss       : {validation_loss:.4f}")
    print(f"Loss gap   : {validation_loss - train_loss:.4f}")
    print(f"Perplexity : {perplexity:.2f}")
    print(f"Top-1      : {top_k_scores[1]:.2%}")
    print(f"Accuracy gap: {train_accuracy - validation_accuracy:.2%}")
    print(f"Top-3      : {top_k_scores[3]:.2%}")
    print(f"Top-5      : {top_k_scores[5]:.2%}")
    print(f"Baseline   : {baseline_accuracy:.2%} (most frequent word)")
    print(f"Random     : {random_accuracy:.2%}")
    print(f"Coverage   : {validation_coverage:.2%} of validation targets")
    print(f"\nEvaluation files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
