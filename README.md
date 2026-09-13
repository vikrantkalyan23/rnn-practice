# Recurrent Neural Networks (RNNs)

RNN, or Recurrent Neural Network, is a neural network architecture designed for sequential data. It processes inputs step by step and maintains a hidden state that carries information from previous time steps. This allows it to use previous context when processing the current input. RNNs are commonly used for tasks such as text processing, speech, and time-series data. However, vanilla RNNs suffer from vanishing and exploding gradient problems, especially with long sequences, which led to architectures such as LSTM and GRU.

``` 
RNN
│
├── Sequential data
│
├── Processes one step at a time
│
├── Hidden state = memory
│
├── Same weights are reused across time
│
├── Trained using Backpropagation Through Time
│
├── Problem → Vanishing/Exploding gradients
│
└── Improvements → LSTM / GRU
```

## RNN vs LSTM vs GRU

| Feature           | RNN     | LSTM         | GRU                      |
| ----------------- | ------- | ------------ | ------------------------ |
| Memory            | Short   | Long         | Long                     |
| Architecture      | Simple  | Complex      | Medium                   |
| Gates             | No      | Yes          | Yes                      |
| Long dependencies | Poor    | Good         | Good                     |
| Parameters        | Few     | More         | Fewer than LSTM          |
| Training          | Faster  | Slower       | Usually faster than LSTM |
| Use today         | Limited | Still useful | Still useful             |

