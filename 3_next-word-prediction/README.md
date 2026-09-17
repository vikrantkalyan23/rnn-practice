# Next-Word Prediction with LSTM

This beginner-friendly project predicts the next word from a short text context.
It uses an LSTM, a type of recurrent neural network that can remember useful
information from earlier words.

Example:

```text
Input:  machine learning
Output candidates: is, models, uses, ...
```

This is a small educational language model. Its corpus is intentionally compact,
so it demonstrates the complete workflow but cannot produce text like a model
trained on millions of sentences.

## Project Structure

```text
3_next-word-prediction/
├── app/
│   ├── config.py       # Paths and beginner-friendly settings
│   ├── data.py         # Cleaning, tokenization, splitting, sequences
│   ├── network.py      # LSTM model architecture
│   ├── predictor.py    # Top-k prediction and text generation
│   ├── schemas.py      # API request and response formats
│   └── main.py         # FastAPI application
├── data/
│   ├── raw/corpus.txt
│   └── processed/cleaned_corpus.txt
├── models/             # Model, tokenizer, configuration, history
├── outputs/            # Evaluation metrics and graphs
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── tests/              # Small unit tests for the data pipeline
```

The main code is flat on purpose. Start with `app/data.py`, continue to
`app/network.py`, and then read the four scripts in execution order.

## How the Model Works

The training data is converted into examples such as:

```text
Context                    Target
machine                    learning
machine learning           is
machine learning is        useful
```

Short contexts are padded with zeros so every input has the same length:

```text
[0, 0, 0, machine, learning] -> is
```

The model contains:

1. `Embedding`: learns a numeric vector for each word.
2. `LSTM(12)`: reads the ordered word vectors and keeps sequence information.
3. `Dropout(0.50)`: reduces dependence on individual neurons.
4. `Dense + softmax`: returns one probability for every vocabulary word.

The model is intentionally compact because a large network memorizes this small
corpus very quickly. It predicts the 80 most frequent training words. Rare words
can still appear as `<OOV>` context, but they are excluded as prediction targets.
This prevents hundreds of one-example classes from dominating training.

## Leakage-Safe Validation

The corpus contains independent sentences, not a chronological time series.
The pipeline therefore shuffles complete lines with a fixed seed and assigns
whole lines to either training or validation.

The split happens before sliding sequences are generated. As a result, two
overlapping phrases from the same sentence cannot appear on opposite sides of
the split. The tokenizer is also fitted on training text only.

## Setup

Run commands from this directory:

```bash
cd 3_next-word-prediction
uv sync
```

## 1. Prepare the Data

```bash
uv run python scripts/prepare_data.py
```

This cleans the corpus, saves the cleaned text, and prints dataset shapes and a
sample token sequence.

## 2. Train the Model

```bash
uv run python scripts/train.py
```

For a quiet run:

```bash
uv run python scripts/train.py --quiet
```

Useful options:

```bash
uv run python scripts/train.py \
  --epochs 150 \
  --batch-size 8 \
  --sequence-length 5 \
  --seed 11
```

Training uses:

- line-level train/validation separation
- fixed random seeds
- gradient clipping
- a small L2 weight penalty
- dropout
- early stopping with best-weight restoration
- learning-rate reduction when validation loss stalls
- checkpointing of the best validation model

Saved artifacts:

```text
models/next_word_model.keras
models/tokenizer.json
models/config.json
models/training_history.json
```

`config.json` records the model's sequence length. Prediction and the API read
this file, preventing configuration mismatches.

## 3. Evaluate the Model

```bash
uv run python scripts/evaluate.py
```

The script reports:

- validation cross-entropy loss
- perplexity
- top-1 accuracy
- top-3 accuracy
- top-5 accuracy

It creates these files:

```text
outputs/training_history.png
outputs/top_k_accuracy.png
outputs/validation_confidence.png
outputs/baseline_comparison.png
outputs/metrics.json
```

### Training history graph

This contains two panels:

- training loss versus validation loss
- training accuracy versus validation accuracy

A growing gap between training and validation indicates overfitting. The red
point identifies the epoch with the best validation loss.

### Top-k accuracy graph

Top-1 requires the correct word to be the model's first choice. Top-5 counts a
prediction as correct when the real word appears anywhere in the five most
likely choices. Top-k is useful because several next words can be reasonable.

### Validation confidence graph

Each bar is the probability assigned to the real next word. Green means that
word was the model's first choice; red means another word ranked higher. This
reveals predictions that accuracy alone hides.

### Baseline comparison graph

This compares the LSTM with random guessing and always predicting the most
frequent training target. The neural model should beat both before its added
complexity is considered useful.

### Perplexity

Perplexity is calculated from cross-entropy loss:

```text
perplexity = exp(loss)
```

Lower is better. It roughly represents how many word choices the model is
uncertain between. Compare perplexity only on the same vocabulary and dataset.

## 4. Predict Words

```bash
uv run python scripts/predict.py "machine learning" --words 5 --top-k 5
```

The script prints the five most likely next words with probabilities and then
generates five words recursively.

Recursive generation can repeat itself because every generated word becomes
input for the next prediction. A larger and more varied corpus is the most
important improvement for generation quality.

## 5. Run the API

```bash
uv run uvicorn app.main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"machine learning","num_words":5}'
```

## Run Tests

```bash
uv run pytest
```

## Fine-Tuning Guide

Change one setting at a time and compare validation metrics.

| Setting | Try | Main tradeoff |
| --- | --- | --- |
| Sequence length | 3, 5, 8 | Longer context needs more data |
| LSTM units | 32, 48, 64 | More units can overfit |
| Embedding size | 16, 32, 64 | Larger vectors learn more parameters |
| Dropout | 0.1 to 0.4 | Too much can cause underfitting |
| Batch size | 8, 16, 32 | Small batches are noisier but update often |

Do not optimize using validation accuracy alone. Check loss, perplexity, top-k
accuracy, confidence, and generated examples together.

Run the reproducible hyperparameter comparison with:

```bash
uv run python scripts/tune.py
```

The tuner compares six compact configurations across validation seeds `7`,
`11`, and `19`. It selects the lowest mean validation loss and saves:

```text
outputs/hyperparameter_tuning.json
outputs/hyperparameter_tuning.png
```

The stage-two average winner used dropout `0.50` and learning rate `0.0015`.
Across three splits it achieved mean validation loss `3.1398`, mean validation
accuracy `35.86%`, and a mean accuracy gap of `5.59` percentage points. However,
it regressed the production seed-11 result, so the saved model conservatively
keeps learning rate `0.001`.

## Current Evaluation

The tuned model currently reports:

```text
Best epoch                  : 147
Training accuracy           : 46.09%
Validation accuracy         : 33.93%
Accuracy gap                : 12.17 percentage points
Training loss               : 2.4472
Validation loss             : 3.1791
Loss gap                    : 0.7319
Validation perplexity       : 24.03
Top-3 accuracy              : 50.00%
Top-5 accuracy              : 55.36%
Most-frequent-word baseline : 12.50%
Random baseline             : 1.25%
Validation target coverage  : 59.57%
```

The vocabulary limit is an intentional tradeoff. It substantially reduces
overfitting and improves accuracy, but the model cannot predict rare words.
`validation_target_coverage` keeps that limitation visible. Increase vocabulary
only after adding enough examples for the additional words.

These production metrics use seed `11`. Across tuning seeds `7`, `11`, and `19`,
the selected configuration has a smaller mean accuracy gap of `5.55` percentage
points. The difference between one split and the cross-split average demonstrates
why results from this small corpus should not be judged from one split alone.

## Current Limitations

- The corpus is still small for language modeling.
- Many valid next words do not appear after the same context in training.
- Rare words are context-only `<OOV>` tokens and are not prediction targets.
- Greedy generation always chooses the highest-probability word and can repeat.
- Validation scores can vary when the small corpus or split changes.

For a meaningful advanced model, use thousands of diverse sentences, retain a
separate test set, and compare LSTM performance with GRU or a small Transformer.
