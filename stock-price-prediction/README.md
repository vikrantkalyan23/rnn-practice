# Stock Price Prediction using RNN

Data → Cleaning → Visualization → Scaling → Sequences → RNN → Training → Evaluation → Prediction → API/UI

```
uv run python src/data_loader.py
```


```
uv run python src/preprocessing.py
```

```
uv run python src/model.py
```

```
uv run python src/train.py
```

For a non-interactive training run:

```
uv run python src/train.py --epochs 80 --no-plot
```

For quiet repeatable training:

```
uv run python src/train.py --epochs 100 --no-plot --quiet --seed 7
```

```
uv run python src/predict.py
```

```
uv run python src/evaluate.py
```

For evaluation without opening plot windows:

```
uv run python src/evaluate.py --no-plot
```

## Optimization Notes

The model uses:

- chronological train/test split
- scaler fitted only on training data
- test sequences that include the last training window as context
- `SimpleRNN(24, activation="relu")`
- `Dense(12, activation="relu")`
- residual prediction: last known price + learned next-day change
- `Adam(learning_rate=0.0003, clipnorm=1.0)`
- `EarlyStopping`
- `ReduceLROnPlateau`
- `ModelCheckpoint`
- `shuffle=False` during training

Latest local evaluation:

```text
MAE  : $2.38
RMSE : $3.53
MAPE : 1.11%
```

Latest train/validation gap:

```text
Training loss at best epoch   : 0.000132
Validation loss at best epoch : 0.000172
Gap                           : 0.000040
Relative gap                  : 23.16%
```

`src/predict.py` first tries live Yahoo Finance data. If the network is not
available, it falls back to `data/stock_data.csv`.
