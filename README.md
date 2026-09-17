# Recurrent Neural Networks: RNN

RNN, or Recurrent Neural Network, is a neural network architecture designed for sequential data. It processes inputs step by step and maintains a hidden state that carries information from previous time steps. This allows it to use previous context when processing the current input. RNNs are commonly used for tasks such as text processing, speech, and time-series data. However, vanilla RNNs suffer from vanishing and exploding gradient problems, especially with long sequences, which led to architectures such as LSTM and GRU.

## Learning Path

1. Understand sequences and sliding windows.
2. Learn how a hidden state gives an RNN memory.
3. Understand input and output shapes.
4. Build a small RNN with Keras.
5. Learn how backpropagation through time trains it.
6. Prepare time-series data without leakage.
7. Evaluate against validation, test, and baseline data.
8. Study vanishing gradients, LSTM, and GRU.
9. Tune and improve the model using chronological experiments.

## 1. What Is Sequential Data?

Sequential data has an order. Changing that order changes its meaning.

Examples include:

- daily temperatures
- stock prices
- words in a sentence
- audio samples
- sensor readings
- website activity over time

For a normal tabular model, rows are often treated as independent. In a
sequence, the current value can depend on earlier values.

```text
Temperature: 20, 21, 23, 22, 24
Time:         1   2   3   4   5
```

Randomly rearranging these values destroys their time relationship.

## 2. From a Sequence to Training Examples

A model needs input examples and correct answers. A sliding window converts one
long sequence into many supervised learning examples.

For this series:

```text
[20, 21, 22, 23, 24, 25, 26]
```

and a window length of `3`, the examples are:

```text
Input X       Target y
[20, 21, 22]  23
[21, 22, 23]  24
[22, 23, 24]  25
[23, 24, 25]  26
```

Here is a beginner-friendly Python implementation:

```python
import numpy as np


def create_sequences(values, sequence_length):
    inputs = []
    targets = []

    for index in range(sequence_length, len(values)):
        start = index - sequence_length
        inputs.append(values[start:index])
        targets.append(values[index])

    X = np.array(inputs)
    y = np.array(targets)
    return X, y


values = [20, 21, 22, 23, 24, 25, 26]
X, y = create_sequences(values, sequence_length=3)

print(X)
print(y)
```

## 3. What Is an RNN?

An RNN is a neural network designed for ordered data. It reads one time step at
a time while carrying an internal memory called the **hidden state**.

```text
x1 -> [RNN] -> h1
                 |
x2 -> [RNN] -> h2
                 |
x3 -> [RNN] -> h3 -> prediction
```

- `x_t` is the input at time step `t`.
- `h_t` is the hidden state after reading `x_t`.
- The hidden state carries information into the next step.
- The same RNN weights are reused at every step.

This repeated use of the same weights is what makes an RNN different from a
separate neural network for every time position.

## 4. The RNN Equation

A simple RNN updates its hidden state with:

```text
h_t = activation(W_xh * x_t + W_hh * h_(t-1) + b_h)
```

It can produce an output with:

```text
y_t = W_hy * h_t + b_y
```

Meaning of each term:

| Symbol | Meaning |
| --- | --- |
| `x_t` | Current input |
| `h_(t-1)` | Previous hidden state |
| `h_t` | New hidden state |
| `W_xh` | Input-to-hidden learned weights |
| `W_hh` | Hidden-to-hidden learned weights |
| `W_hy` | Hidden-to-output learned weights |
| `b_h`, `b_y` | Learned biases |

The activation function is usually `tanh` in a traditional SimpleRNN. Some
forecasting examples in this repository use `relu` because it can behave better
on their small increasing datasets. Activation choice is an experiment, not a
universal rule.

## 5. A Conceptual RNN Step in NumPy

This example shows one recurrent update. It is for understanding, not training.

```python
import numpy as np

x_t = np.array([[0.5]])
previous_hidden = np.array([[0.0, 0.0]])

input_weights = np.array([[0.4, -0.2]])
recurrent_weights = np.array([
    [0.1, 0.3],
    [-0.4, 0.2],
])
bias = np.array([[0.0, 0.0]])

hidden = np.tanh(
    x_t @ input_weights
    + previous_hidden @ recurrent_weights
    + bias
)

print(hidden)
```

During a complete sequence, the new `hidden` value becomes
`previous_hidden` for the next step.

## 6. Understanding RNN Shapes

Keras recurrent layers expect a three-dimensional input:

```text
(samples, time steps, features)
```

Example:

```text
(32, 60, 1)
  |   |  |
  |   |  +-- one feature per day
  |   +----- 60 time steps in each sequence
  +--------- 32 sequences in this batch
```

Common terms:

- **Sample**: one complete input sequence
- **Time step**: one position in that sequence
- **Feature**: one measurement at a time step
- **Batch**: several samples processed before a weight update

For closing price and volume together, the shape could be `(32, 60, 2)` because
each day has two features.

To add the feature dimension to simple one-dimensional sequences:

```python
X = X.reshape(X.shape[0], X.shape[1], 1)
```

## 7. First RNN with Keras

This model reads five temperatures and predicts the next one:

```python
from tensorflow.keras.layers import Dense, Input, SimpleRNN
from tensorflow.keras.models import Sequential

model = Sequential([
    Input(shape=(5, 1)),
    SimpleRNN(16, activation="tanh"),
    Dense(1),
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"],
)

model.summary()
```

Layer roles:

1. `Input(shape=(5, 1))` accepts five time steps with one feature each.
2. `SimpleRNN(16)` creates a hidden state with 16 values.
3. `Dense(1)` converts the hidden state into one prediction.

Train it with:

```python
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_validation, y_validation),
    epochs=100,
    batch_size=32,
    shuffle=False,
)
```

For time-series forecasting, `shuffle=False` keeps samples in chronological
order during each epoch.

## 8. Many-to-One and Many-to-Many RNNs

RNN output structure depends on the task.

### Many-to-one

Use a complete sequence to produce one result.

```text
60 prices -> next price
sentence  -> sentiment label
```

Keras uses only the final RNN output by default:

```python
SimpleRNN(32)
```

### Many-to-many

Produce an output for every time step.

```text
word sequence -> tag for every word
sensor values -> anomaly score at every step
```

Ask Keras to return the full output sequence:

```python
SimpleRNN(32, return_sequences=True)
```

Stacked recurrent layers also require earlier layers to return sequences:

```python
model = Sequential([
    Input(shape=(60, 1)),
    SimpleRNN(32, return_sequences=True),
    SimpleRNN(16),
    Dense(1),
])
```

## 9. How an RNN Learns

Training repeats the following cycle:

1. **Forward pass**: process each sequence and make predictions.
2. **Loss calculation**: compare predictions with correct targets.
3. **Backpropagation through time**: calculate how every reused weight affected
   the loss across the sequence.
4. **Optimization**: update weights to reduce future error.

### Mean squared error

Regression models commonly use mean squared error (MSE):

```text
MSE = average((actual - predicted)^2)
```

Squaring makes large errors count more heavily.

### Learning rate

The optimizer's learning rate controls weight-update size:

- Too high: loss may jump or diverge.
- Too low: learning may be very slow.
- Reasonable: loss generally improves and settles.

### Epoch and batch

- **Epoch**: one pass through all training examples
- **Batch size**: examples used for one optimizer update

More epochs do not guarantee a better model. Validation performance determines
when useful learning has stopped.

## 10. Backpropagation Through Time

An RNN can be pictured as the same cell unrolled across time:

```text
       shared weights      shared weights
x1 -> cell 1 -> h1 -> cell 2 -> h2 -> cell 3 -> h3
                       ^ x2                ^ x3
```

Backpropagation follows this graph backward. A weight used at every time step
receives gradient information from many positions.

For long sequences, repeatedly multiplying gradients can cause two problems:

- **Vanishing gradients**: gradients approach zero, so early inputs are barely
  learned.
- **Exploding gradients**: gradients become extremely large and training turns
  unstable.

Gradient clipping can limit exploding gradients:

```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(
    learning_rate=0.001,
    clipnorm=1.0,
)
```

LSTM and GRU were designed mainly to improve learning across longer
dependencies.

## 11. RNN vs LSTM vs GRU

| Feature | SimpleRNN | LSTM | GRU |
| --- | --- | --- | --- |
| Main memory | Hidden state | Hidden and cell state | Hidden state |
| Gates | None | Input, forget, output | Update, reset |
| Parameter count | Lowest | Highest | Usually between RNN and LSTM |
| Long dependencies | Weak | Stronger | Stronger |
| Training speed | Usually fastest | Usually slowest | Often faster than LSTM |
| Good first use | Short/simple patterns | Longer complex patterns | Efficient long-memory model |

These are general tendencies. The best architecture depends on the dataset and
must be measured on unseen data.

## 12. LSTM: Long Short-Term Memory

An LSTM adds a cell state and three gates:

- **Forget gate**: decides what old information to remove.
- **Input gate**: decides what new information to store.
- **Output gate**: decides what memory to expose as output.

The cell state creates a controlled path for information and gradients across
many time steps.

```python
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Sequential

model = Sequential([
    Input(shape=(60, 1)),
    LSTM(32),
    Dense(1),
])
```

Use an LSTM when the target may depend on events much earlier in the sequence,
but always compare it with a simpler baseline.

## 13. GRU: Gated Recurrent Unit

A GRU uses update and reset gates. It has no separate cell state and usually has
fewer parameters than an equivalent LSTM.

```python
from tensorflow.keras.layers import Dense, GRU, Input
from tensorflow.keras.models import Sequential

model = Sequential([
    Input(shape=(60, 1)),
    GRU(32),
    Dense(1),
])
```

GRU is a practical choice when you want gated memory with a simpler structure
than LSTM.

## 14. Preparing Time-Series Data Correctly

A robust forecasting pipeline follows chronological order:

```text
oldest                                                    newest
|---------------- train ----------------|-- validation --|--- test ---|
```

- **Training set**: updates model weights.
- **Validation set**: selects settings and stopping epoch.
- **Test set**: measures final performance once decisions are complete.

Never randomly split ordinary forecasting data. A random split can place future
observations in training and earlier observations in testing.

## 15. Scaling Without Data Leakage

Neural networks often train more reliably when features have similar ranges.
For example, `MinMaxScaler` can map training values near the range 0 to 1.

Correct order:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_values)
validation_scaled = scaler.transform(validation_values)
test_scaled = scaler.transform(test_values)
```

Incorrect order:

```python
# Wrong: the scaler learns information from future data.
all_scaled = scaler.fit_transform(all_values)
```

The validation and test values may scale below 0 or above 1 when they fall
outside the training range. That is not automatically an error; it reveals that
future data differs from training data.

Save the fitted scaler with the model. Prediction must use the exact same
transformation used during training.

## 16. Leakage Checklist

Data leakage occurs when training receives information that would not be
available at real prediction time.

Check all of these:

- Split chronologically before fitting transformations.
- Fit scalers and feature engineering only on training data.
- Do not use future prices inside input windows.
- Do not tune hyperparameters on the test set.
- Do not calculate centered rolling averages that include future rows.
- Do not accidentally use the target itself as an input feature.
- Keep the final test period untouched until final evaluation.

Using the last training window as past context for the first validation or test
prediction is valid. Those values occurred before the target and would be known
at prediction time.

## 17. Training, Validation, and Test Loss

### Underfitting

Both training and validation loss remain high. The model has not learned the
pattern well enough.

Possible responses:

- train longer if loss is still improving
- increase model capacity carefully
- improve input features
- verify preprocessing and target alignment

### Overfitting

Training loss falls while validation loss stops improving or rises. The model
is fitting training-specific patterns.

Possible responses:

- use early stopping
- reduce model size
- add a small amount of dropout or weight regularization
- collect more representative data
- simplify noisy input features

### Healthy training

Training and validation loss improve, then stabilize with a reasonable gap.
A small gap is expected because the model directly learns from training data.

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=0.00001,
    ),
]
```

## 18. Evaluation Metrics

Do not judge a regression model only from a graph.

### MAE

Mean absolute error is the average absolute difference:

```text
MAE = average(|actual - predicted|)
```

It uses the target's original unit after inverse scaling, so a stock MAE of
`2.50` means an average error of about `$2.50`.

### RMSE

Root mean squared error penalizes large errors more strongly:

```text
RMSE = sqrt(average((actual - predicted)^2))
```

### MAPE

Mean absolute percentage error expresses average error as a percentage. It can
be misleading when actual values are zero or close to zero.

### Direction accuracy

For some financial experiments, you may also measure whether the predicted
direction matches the actual direction. Good price MAE does not guarantee good
direction accuracy.

## 19. Always Use a Baseline

A neural network is useful only if it improves on a simple method.

For one-step time-series forecasting, a strong naive baseline is:

```text
next predicted value = latest known value
```

```python
naive_predictions = X_test[:, -1, 0]
```

Compare the model and baseline on exactly the same targets and scale. Stock
prices are highly autocorrelated, so a visually close RNN line may simply behave
like this naive forecast.

Other baselines include:

- historical mean
- moving average
- seasonal value from the previous cycle
- linear regression

## 20. One-Step and Multi-Step Forecasting

### One-step forecast

Predict only the next value using real recent history.

```text
[real t-59 ... real t] -> predict t+1
```

### Direct multi-output forecast

Predict several future values at once:

```python
model.add(Dense(7))  # Predict the next seven values.
```

### Recursive forecast

Predict one step, append that prediction, and use it to predict again.

```text
history -> prediction 1
history + prediction 1 -> prediction 2
```

Recursive errors accumulate because later predictions depend on earlier
predictions rather than real observations.

## 21. Advanced Architecture Options

### Stacked recurrent layers

```python
model = Sequential([
    Input(shape=(60, 4)),
    GRU(64, return_sequences=True),
    GRU(32),
    Dense(1),
])
```

This adds capacity but also increases overfitting risk.

### Bidirectional RNN

A bidirectional layer reads a sequence forward and backward:

```python
from tensorflow.keras.layers import Bidirectional, LSTM

Bidirectional(LSTM(32))
```

It is useful when the entire input sequence is available, such as sentence
classification. Be careful in forecasting: it must never read observations
after the prediction cutoff.

### Residual prediction

Instead of predicting the complete next value, predict its change from the
latest value:

```text
prediction = latest value + learned change
```

The stock project uses this approach. It provides a sensible starting point for
slow-moving series, but makes baseline comparison especially important.

### Regularization

Options include:

- dropout between layers
- recurrent dropout inside a recurrent layer
- L1 or L2 weight penalties
- smaller layers
- early stopping

Apply regularization only after measuring a real generalization problem.

## 22. Features and Targets

A univariate model uses one feature, such as closing price. A multivariate model
can use several values per time step:

```text
[close, volume, return, moving average]
```

Potential time-series features include:

- lagged values
- percentage or log returns
- rolling mean and standard deviation
- calendar information
- related sensor or market variables

Every feature must be computable using information available at prediction
time. Compute rolling and aggregate features carefully to avoid future leakage.

Predicting returns instead of raw prices can reduce trend dependence, but
returns are often noisier and more difficult to forecast.

## 23. Walk-Forward Validation

A single validation period may give a result that depends heavily on one market
or weather regime. Walk-forward validation tests several chronological periods:

```text
Fold 1: [train------][validate]
Fold 2: [train------------][validate]
Fold 3: [train------------------][validate]
```

For each fold:

1. Fit preprocessing on that fold's training period.
2. Train a new model.
3. Evaluate on the following validation period.
4. Record both model and baseline metrics.

Average performance and variation across folds provide stronger evidence than
one lucky split.

## 24. Hyperparameter Tuning

Useful hyperparameters include:

- sequence length
- RNN type: SimpleRNN, LSTM, or GRU
- number of recurrent units
- number of layers
- learning rate
- batch size
- activation function
- dropout or L2 strength

Change one factor at a time when learning. Keep the chronological split, seed,
metric, and baseline fixed so comparisons remain meaningful.

Never select settings from test performance. Use training and validation data
for decisions, then use the test set once for the final estimate.

## 25. Reproducibility

Seeds reduce randomness but may not make every hardware configuration perfectly
identical.

```python
import random
import numpy as np
import tensorflow as tf

SEED = 11

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

Also record:

- dataset version and date range
- train, validation, and test boundaries
- scaler and feature definitions
- sequence length
- architecture and optimizer settings
- best epoch and evaluation metrics
- software versions

## 26. Common Problems

### Predictions are too smooth

MSE encourages the model to predict central values when the future is uncertain.
The model may also lack useful features for sudden changes.

### Predictions lag behind actual values

The model may behave like a moving average or naive forecast. Compare it with
the naive baseline and consider returns, additional causal features, or a
different target.

### Training loss jumps

Try a lower learning rate, gradient clipping, scaled features, or a smaller
network. Also check for missing or extreme values.

### Validation is much worse than training

Possible causes include overfitting, a changed future data distribution,
leakage in experiment selection, or preprocessing differences.

### Results change every run

Set seeds, use the same split, and record configuration. Small datasets can
still produce unstable results, so compare several seeds or validation folds.

### Output goes outside the expected range

A final linear `Dense(1)` layer is allowed to output any value. This is often
correct for regression. Check scaling, inverse scaling, target alignment, and
whether the future lies outside the training distribution.

## 27. Repository Demos

### Temperature prediction: first project

This project is best for learning sequence creation, scaling, SimpleRNN, and
one-step prediction.

```bash
cd temperature-prediction
uv run src/prepare_data.py
uv run src/model.py
uv run src/train.py --epochs 60
uv run src/evaluate.py --no-plot
uv run src/predict.py 45 46 47 48 49
```

See the full guide in
[`temperature-prediction/README.md`](temperature-prediction/README.md).

### Stock price prediction: complete project

This project demonstrates chronological splitting, train-only scaling, a
residual RNN, callbacks, test metrics, plots, online prediction, and FastAPI.

```bash
cd stock-price-prediction
uv sync
uv run python src/train.py --epochs 100 --no-plot --quiet --seed 11
uv run python src/evaluate.py --no-plot
uv run python src/predict.py
uv run uvicorn app.main:app --reload
```

The interactive notebook is at
[`stock-price-prediction/notebook/exploration.ipynb`](stock-price-prediction/notebook/exploration.ipynb).

See the full project guide in
[`stock-price-prediction/README.md`](stock-price-prediction/README.md).

## 28. Suggested Practice Exercises

1. Change the temperature sequence length from 5 to 7 and compare MAE.
2. Change `SimpleRNN` units and record training and validation loss.
3. Replace `SimpleRNN` with `GRU`, keeping all other settings fixed.
4. Replace `GRU` with `LSTM` and compare parameter counts.
5. Add a naive baseline to the temperature project.
6. Plot residuals: `actual - predicted`.
7. Measure stock direction accuracy.
8. Run the stock model with three fixed seeds and compare variation.
9. Implement walk-forward validation.
10. Add a second feature and verify that its calculation has no leakage.

## 29. Model Selection Checklist

Before calling an RNN successful, verify:

- Data is sorted chronologically.
- Train, validation, and test periods do not overlap incorrectly.
- Scalers are fitted only on training data.
- Every input feature is available at prediction time.
- Input and target windows are aligned correctly.
- Training uses a suitable loss and stable learning rate.
- The best validation weights are restored.
- Metrics are calculated after inverse scaling.
- The model is compared with a naive baseline.
- Results are stable across more than one period or seed.
- The final test set was not used for tuning.

## 30. Final Mental Model

An RNN reads a sequence one item at a time. At each step, it combines the current
input with its existing hidden state to create a new hidden state. Training uses
backpropagation through time to adjust the shared weights. LSTM and GRU add gates
that help preserve useful information over longer sequences.

For real forecasting, architecture is only one part of the problem. Correct
time ordering, leakage-free preprocessing, honest baselines, chronological
validation, and clear metrics usually matter more than adding another layer.
