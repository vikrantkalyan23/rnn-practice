# Temperature Prediction Using an RNN

This is a small beginner-friendly project that predicts the next temperature
from the previous few temperatures.

The project uses:

- Python
- NumPy and Pandas for data handling
- TensorFlow/Keras for the RNN model
- Matplotlib for the evaluation chart

## Project Structure

```text
temperature-prediction/
  data/
    temperature.csv          # The sample temperature data
  models/
    temperature_rnn.keras    # Saved trained model
    temperature_scaler.npz   # Saved scaling values
  src/
    data_utils.py            # Simple shared helper functions
    prepare_data.py          # Shows how data becomes training examples
    model.py                 # Builds the RNN model
    train.py                 # Trains and saves the model
    evaluate.py              # Compares predictions with real values
    predict.py               # Predicts the next temperature
```

## Setup

Create a virtual environment:

```bash
uv venv --python 3.12
```

Install dependencies:

```bash
uv pip install -r requirements.txt
```

Check TensorFlow:

```bash
uv run python -c "import tensorflow as tf; print(tf.__version__)"
```

## Run the App

Run these commands from the `temperature-prediction` folder.

### 1. See how the data is prepared

```bash
uv run src/prepare_data.py
```

This shows examples like:

```text
Input:  20, 21, 22, 23, 24
Target: 25
```

The RNN learns from many examples like that.

### 2. View the model

```bash
uv run src/model.py
```

This prints the layers in the neural network.

### 3. Train the model

```bash
uv run src/train.py
```

For a faster beginner test, run fewer epochs:

```bash
uv run src/train.py --epochs 60
```

Training saves two files:

- `models/temperature_rnn.keras`: the trained neural network
- `models/temperature_scaler.npz`: scaling values used by the model

### 4. Evaluate the model

```bash
uv run src/evaluate.py
```

This prints each real temperature next to the model prediction and opens a
chart.

For terminal-only checking without a chart:

```bash
uv run src/evaluate.py --no-plot
```

### 5. Predict the next temperature

Use the default example:

```bash
uv run src/predict.py
```

Or pass your own recent temperatures:

```bash
uv run src/predict.py 45 46 47 48 49
```

The number of temperatures must match the sequence length used during training.
The default sequence length is `5`.

## Important Beginner Concepts

### What is a sequence?

A sequence is a group of recent values. With a sequence length of `5`, the app
uses five temperatures to predict the next one:

```text
[20, 21, 22, 23, 24] -> 25
[21, 22, 23, 24, 25] -> 26
[22, 23, 24, 25, 26] -> 27
```

### Why scale the data?

Neural networks usually train better when numbers are close to zero. The app
scales temperatures before training, then converts predictions back to normal
temperature values before printing them.

### What is MAE?

`MAE` means mean absolute error. It is the average difference between the real
temperature and the predicted temperature.

Lower MAE is better.

## Common Changes

Change training length:

```bash
uv run src/train.py --epochs 100
```

Change how many previous days are used:

```bash
uv run src/train.py --sequence-length 7
```

Then predict with the same number of values:

```bash
uv run src/predict.py 45 46 47 48 49 50 51
```

Change the RNN layer size:

```bash
uv run src/train.py --rnn-units 16
```

