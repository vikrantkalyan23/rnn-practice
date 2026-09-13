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
- `SimpleRNN(64, activation="relu")`
- `Dense(16, activation="relu")`
- `EarlyStopping`
- `ReduceLROnPlateau`
- `ModelCheckpoint`
- `shuffle=False` during training

Latest local evaluation:

```text
MAE  : $2.87
RMSE : $4.10
MAPE : 1.34%
```

`src/predict.py` first tries live Yahoo Finance data. If the network is not
available, it falls back to `data/stock_data.csv`.
