# PTB LSTM model with TensorFlow Estimators

This is a reimplementation of the code available in [tensorflow/tutorials/rnn/ptb](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb), using [Estimators](https://www.tensorflow.org/versions/master/programmers_guide/estimators) high level API.

## Notes of the implementation

* Tested with TF 1.6.0 and Python 3.6.3

* Feeding RNN final states for the next batch using a custom hook (as suggested in [StackOverflow](https://stackoverflow.com/q/46613594))

* Using newer [tf.data](https://www.tensorflow.org/versions/master/api_docs/python/tf/data) API for feeding data and cleaner [tf.layers](https://www.tensorflow.org/versions/master/api_docs/python/tf/layers) API for building the network model

* cuDNN RNN implementation, 16-bit float and multi-GPU options have been omitted for simplicity

## Results

Results may vary to some extent due to the random initialization.

### Small

The training speed for both the Estimator and original implementations are 44 steps/sec (~17600 words per second) for a K80.

| Epoch | Estimator | Original |
|:-----:|:---------:|:--------:|
| 1 | Train: 269.615<br>Valid: 176.892 | Train: 268.352<br>Valid: 181.622 |
| 5 | Train: 65.613<br>Valid: 118.120 | Train: 65.550<br>Valid: 119.557 |
| 10 | Train: 41.456<br>Valid: 119.530 | Train: 41.478<br>Valid: 121.326 |
| 13 | Train: 40.530<br>Valid: 118.812<br>Test: 114.159 | Train: 40.551<br>Valid: 120.697<br>Test: 114.932 |

### Medium

The training speed for both the Estimator and original implementations are 8 steps/sec (~5600 words per second) for a K80.

| Epoch | Estimator | Original |
|:-----:|:---------:|:--------:|
| 1 | Train: 361.183<br>Valid: 206.924 | Train: 361.598<br>Valid: 203.769 |
| 10 | Train: 63.409<br>Valid: 90.755 | Train: 63.650<br>Valid: 91.271 |
| 20 | Train: 47.341<br>Valid: 87.994 | Train: 47.685<br>Valid: 88.572 |
| 30 | Train: 45.799<br>Valid: 87.394 | Train: 46.056<br>Valid: 88.256 |
| 39 | Train: 45.522<br>Valid: 87.337<br>Test: 83.731 | Train: 45.865<br>Valid: 88.054<br>Test: 83.695 |
