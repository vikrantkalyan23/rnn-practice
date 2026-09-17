import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # noqa: E402

from app.config import DATA_PATH, OUTPUT_DIR  # noqa: E402
from app.data import (  # noqa: E402
    count_possible_targets,
    create_sequences,
    create_tokenizer,
    get_vocabulary_size,
    split_text_train_validation_test,
)
from app.network import build_model  # noqa: E402


CONFIG_PATH = PROJECT_DIR / "experiments" / "tuning_config.json"
JSON_REPORT_PATH = OUTPUT_DIR / "hyperparameter_tuning.json"
CSV_REPORT_PATH = OUTPUT_DIR / "hyperparameter_tuning.csv"
PLOT_PATH = OUTPUT_DIR / "hyperparameter_tuning.png"

CONFIG_FIELDS = (
    "learning_rate",
    "batch_size",
    "embedding_dim",
    "lstm_units",
    "sequence_length",
    "dropout",
    "l2_strength",
    "max_vocab_size",
)
METRICS = (
    "best_epoch",
    "train_loss",
    "validation_loss",
    "loss_gap",
    "train_accuracy",
    "top_1_accuracy",
    "top_3_accuracy",
    "top_5_accuracy",
    "validation_target_coverage",
)


def load_experiment_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    baseline = config["baseline"]
    candidates = []
    for experiment in config["experiments"]:
        candidate = {**baseline, **experiment}
        missing = [field for field in CONFIG_FIELDS if field not in candidate]
        if missing:
            raise ValueError(f"{candidate.get('name', 'experiment')} missing {missing}")
        candidates.append(candidate)

    names = [candidate["name"] for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("Experiment names must be unique.")
    return config, candidates


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def top_k_accuracy(probabilities, targets, k):
    k = min(k, probabilities.shape[1])
    top_ids = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top_ids == targets[:, None], axis=1)))


def evaluate_candidate(candidate, text, protocol):
    folds = []
    for seed in protocol["validation_seeds"]:
        tf.keras.backend.clear_session()
        set_random_seed(seed)
        train_text, validation_text, test_text = split_text_train_validation_test(
            text,
            validation_ratio=protocol["validation_ratio"],
            test_ratio=protocol["test_ratio"],
            test_seed=protocol["test_split_seed"],
            validation_seed=seed,
        )
        tokenizer = create_tokenizer(train_text, candidate["max_vocab_size"])
        X_train, y_train = create_sequences(
            train_text, tokenizer, candidate["sequence_length"]
        )
        X_validation, y_validation = create_sequences(
            validation_text, tokenizer, candidate["sequence_length"]
        )
        model = build_model(
            vocab_size=get_vocabulary_size(tokenizer),
            sequence_length=candidate["sequence_length"],
            embedding_dim=candidate["embedding_dim"],
            lstm_units=candidate["lstm_units"],
            dropout=candidate["dropout"],
            learning_rate=candidate["learning_rate"],
            l2_strength=candidate["l2_strength"],
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_validation, y_validation),
            epochs=protocol["max_epochs"],
            batch_size=candidate["batch_size"],
            shuffle=False,
            verbose=0,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=protocol["early_stopping_patience"],
                    min_delta=protocol["early_stopping_min_delta"],
                    restore_best_weights=True,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=protocol["lr_reduction_factor"],
                    patience=protocol["lr_reduction_patience"],
                    min_lr=protocol["minimum_learning_rate"],
                ),
            ],
        )
        best_index = int(np.argmin(history.history["val_loss"]))
        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        validation_loss, validation_accuracy = model.evaluate(
            X_validation, y_validation, verbose=0
        )
        probabilities = model.predict(X_validation, verbose=0)
        folds.append(
            {
                "validation_seed": seed,
                "best_epoch": best_index + 1,
                "epochs_ran": len(history.history["loss"]),
                "train_examples": len(y_train),
                "validation_examples": len(y_validation),
                "fixed_test_lines": len(test_text.splitlines()),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "loss_gap": float(validation_loss - train_loss),
                "train_accuracy": float(train_accuracy),
                "top_1_accuracy": float(validation_accuracy),
                "top_3_accuracy": top_k_accuracy(probabilities, y_validation, 3),
                "top_5_accuracy": top_k_accuracy(probabilities, y_validation, 5),
                "validation_target_coverage": len(y_validation)
                / count_possible_targets(validation_text),
            }
        )

    result = {field: candidate[field] for field in ("name", *CONFIG_FIELDS)}
    result["folds"] = folds
    for metric in METRICS:
        values = [fold[metric] for fold in folds]
        result[f"mean_{metric}"] = float(np.mean(values))
        result[f"std_{metric}"] = float(np.std(values))
    return result


def write_csv(results):
    fieldnames = [
        "run",
        "name",
        *CONFIG_FIELDS,
        "validation_seed",
        "best_epoch",
        "epochs_ran",
        "train_examples",
        "validation_examples",
        "fixed_test_lines",
        "train_loss",
        "validation_loss",
        "loss_gap",
        "train_accuracy",
        "top_1_accuracy",
        "top_3_accuracy",
        "top_5_accuracy",
        "validation_target_coverage",
    ]
    with CSV_REPORT_PATH.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        run = 0
        for result in results:
            for fold in result["folds"]:
                run += 1
                writer.writerow(
                    {
                        "run": run,
                        "name": result["name"],
                        **{field: result[field] for field in CONFIG_FIELDS},
                        **fold,
                    }
                )


def write_report(config, results, status):
    ranked = sorted(results, key=lambda result: result["mean_validation_loss"])
    report = {
        "status": status,
        "selection_metric": "lowest mean validation loss across fixed validation splits",
        "protocol": config["protocol"],
        "test_partition_used_for_tuning": False,
        "completed_experiments": len(results),
        "best_candidate": ranked[0]["name"] if ranked else None,
        "results": ranked,
    }
    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(results)


def plot_results(results):
    ranked = sorted(results, key=lambda result: result["mean_validation_loss"])
    names = [result["name"] for result in ranked]
    losses = [result["mean_validation_loss"] for result in ranked]
    top_1 = [result["mean_top_1_accuracy"] for result in ranked]
    top_3 = [result["mean_top_3_accuracy"] for result in ranked]
    top_5 = [result["mean_top_5_accuracy"] for result in ranked]

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(names, losses, color="#2878B5")
    axes[0].invert_yaxis()
    axes[0].set_title("Mean Validation Loss")
    axes[0].set_xlabel("Cross-entropy loss")
    axes[0].grid(axis="x", alpha=0.25)

    positions = np.arange(len(names))
    width = 0.25
    axes[1].barh(positions - width, top_1, width, label="Top-1")
    axes[1].barh(positions, top_3, width, label="Top-3")
    axes[1].barh(positions + width, top_5, width, label="Top-5")
    axes[1].set_yticks(positions, names)
    axes[1].invert_yaxis()
    axes[1].set_title("Mean Top-K Accuracy")
    axes[1].set_xlabel("Accuracy")
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.25)

    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=160)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config, candidates = load_experiment_config()
    text = DATA_PATH.read_text(encoding="utf-8")
    config["protocol"]["corpus_test_protocol_fingerprint"] = hashlib.sha256(
        f"{config['protocol']['test_split_seed']}:{text}".encode()
    ).hexdigest()
    results = []

    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] {candidate['name']}")
        result = evaluate_candidate(candidate, text, config["protocol"])
        results.append(result)
        write_report(config, results, status="running")
        print(
            f"  loss={result['mean_validation_loss']:.4f}, "
            f"top1={result['mean_top_1_accuracy']:.2%}, "
            f"top3={result['mean_top_3_accuracy']:.2%}, "
            f"top5={result['mean_top_5_accuracy']:.2%}"
        )

    write_report(config, results, status="complete")
    plot_results(results)
    best = min(results, key=lambda result: result["mean_validation_loss"])
    print(f"\nBest candidate: {best['name']}")
    print(f"Mean validation loss: {best['mean_validation_loss']:.4f}")
    print(f"Mean Top-1: {best['mean_top_1_accuracy']:.2%}")
    print(f"JSON report: {JSON_REPORT_PATH}")
    print(f"CSV ledger : {CSV_REPORT_PATH}")


if __name__ == "__main__":
    main()
