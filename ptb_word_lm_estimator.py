"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of BASIC, or BLOCK,
             representing basic_lstm, and lstm_block_cell classes.

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz"""

import argparse
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import nest

import reader

BASIC = "basic"
BLOCK = "block"


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class PrintingHook(tf.train.SessionRunHook):
    """Hook that emulates the printing information in the original
    PTB tensorflow implementation."""

    def __init__(self, tensors):
        self.cost = tensors["cost"]
        self.step = tensors["global_step"]
        self.params = tensors["params"]
        self.training = tensors["training"]
        self.costs = 0.0
        self.iters = 0

    def before_run(self, run_context):
        self.start_time = time.time()
        run_args = tf.train.SessionRunArgs([self.cost, self.step])
        return run_args

    def after_run(self, run_context, run_values):
        step = run_values.results[1]
        self.costs += run_values.results[0]
        self.iters += self.params.num_steps
        if (step % (self.params.epoch_size // 10) == 10 and self.training):
            print("Step: %.0f Epoch: %.3f perplexity: %.3f speed: %.0f wps" %
                  (step, step * 1.0 / self.params.epoch_size,
                   np.exp(self.costs / self.iters),
                   self.params.num_steps * self.params.batch_size /
                   (time.time() - self.start_time)))

    def end(self, session):
        if self.training:
            split = "Train"
        elif not self.training and self.params.batch_size == 1:
            split = "Test"
        else:
            split = "Valid"
        print("Epoch: %d %s Perplexity: %.3f" %
              (int(session.run(self.step) / self.params.epoch_size), split,
               np.exp(self.costs / self.iters)))


class RNNStateHook(tf.train.SessionRunHook):
    """Hook for feeding final states of the previous batch as initial state."""

    def __init__(self, tensors):
        self.initial_state = tensors["initial_state"]
        self.output_state = tensors["output_state"]
        self.current_state = tf.get_collection('rnn_input_state')

    def after_create_session(self, session, coord):
        self.current_state = [np.zeros(var.shape) for var in self.current_state]

    def before_run(self, run_context):
        run_args = tf.train.SessionRunArgs(
            self.output_state, {self.initial_state: self.current_state})
        return run_args

    def after_run(self, run_context, run_values):
        self.current_state = run_values.results


class PTBInput(object):
    """The input data."""
    def __init__(self, params, data, name=None):
        self.batch_size = batch_size = params.batch_size
        self.num_steps = num_steps = params.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data = lambda: reader.ptb_producer(
            data, params.batch_size, params.num_steps, name)


class PTBModel(object):
    """The PTB model."""

    def __init__(self, params, training):
        self.training = training
        self.params = params

    def __call__(self, inputs):
        end_points = {}
        # Embedding layer
        with tf.device("/cpu:0"):
            inputs = tf.contrib.layers.embed_sequence(
                inputs,
                vocab_size=self.params.vocab_size,
                embed_dim=self.params.hidden_size,
                scope="embedding")
            end_points["inputs"] = inputs

        if self.params.keep_prob < 1 and self.training:
            inputs = tf.layers.dropout(
                inputs, rate=self.params.keep_prob, training=self.training)

        # RNN graph
        cell = tf.contrib.rnn.MultiRNNCell(
            [self.make_cell() for _ in range(self.params.num_layers)],
            state_is_tuple=True)
        end_points["cell"] = cell

        initial_state = cell.zero_state(self.params.batch_size, tf.float32)
        end_points["initial_state"] = initial_state

        for tensor in nest.flatten(initial_state):
            tf.add_to_collection('rnn_input_state', tensor)

        inputs = tf.unstack(inputs, num=self.params.num_steps, axis=1)
        outputs, state = tf.nn.static_rnn(cell, inputs,
                                          initial_state=initial_state)
        output = tf.reshape(tf.concat(outputs, 1),
                            [-1, self.params.hidden_size])
        end_points["output"] = output
        end_points["output_state"] = state

        logits = tf.layers.dense(output, self.params.vocab_size)
        logits = tf.reshape(logits,
                            [self.params.batch_size, self.params.num_steps,
                             self.params.vocab_size])
        end_points["logits"] = logits
        return logits, end_points

    def make_cell(self):
        """Cell specification for the RNN."""
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        if self.params.rnn_mode == BASIC:
            cell = tf.contrib.rnn.BasicLSTMCell(
                self.params.hidden_size, forget_bias=0.0, state_is_tuple=True)
        elif self.params.rnn_mode == BLOCK:
            cell = tf.contrib.rnn.LSTMBlockCell(
                self.params.hidden_size, forget_bias=0.0)
        else:
            raise ValueError("rnn_mode %s not supported" % self.params.rnn_mode)

        if self.params.keep_prob < 1 and self.training:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=self.params.keep_prob)
        return cell


def model_fn(features, labels, mode, params):
    """Function to build the PTB model."""

    training = (mode == tf.estimator.ModeKeys.TRAIN)
    if not training and (features.get_shape()[0] == 1):
        params.batch_size = 1
        params.num_steps = 1

    global_step = tf.train.get_global_step()
    init = tf.random_uniform_initializer(-params.init_scale, params.init_scale)

    with tf.variable_scope("Model", reuse=None, initializer=init):
        model = PTBModel(params, training=training)
        logits, end_points = model(features)

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        labels,
        tf.ones([params.batch_size, params.num_steps]),
        average_across_timesteps=False,
        average_across_batch=True)
    loss = tf.reduce_sum(loss)

    # Build a learning rate decay function to use with optimize_loss
    lr_decay_fn = lambda lr, gs: lr * tf.pow(
        params.lr_decay,
        tf.to_float(tf.maximum(
            tf.to_int32(gs / params.epoch_size) + 1 - params.max_epoch, 0)))

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        optimizer=tf.train.GradientDescentOptimizer,
        learning_rate=params.learning_rate,
        learning_rate_decay_fn=lr_decay_fn,
        clip_gradients=float(params.max_grad_norm))

    hook_list = []
    tensors = {"cost": loss, "global_step": global_step, "params": params,
               "training": training}
    hook_list.append(RNNStateHook(tensors=end_points))
    hook_list.append(PrintingHook(tensors=tensors))

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_hooks=hook_list,
        evaluation_hooks=hook_list)


def main(_):

    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, vocab_len = raw_data

    def get_config():
        """Returns the model config required according to FLAGS.model."""
        if FLAGS.model == "small":
            return SmallConfig()
        elif FLAGS.model == "medium":
            return MediumConfig()
        elif FLAGS.model == "large":
            return LargeConfig()
        elif FLAGS.model == "test":
            return TestConfig()
        else:
            raise ValueError("Invalid model: %s" % FLAGS.model)

    config = get_config()
    config.vocab_size = vocab_len

    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    # Train and eval input functions
    train_input = PTBInput(params=config, data=train_data, name="TrainInput")
    config.epoch_size = train_input.epoch_size
    valid_input = PTBInput(params=config, data=valid_data, name="ValidInput")
    test_input = PTBInput(params=eval_config, data=test_data, name="TestInput")

    model_function = model_fn

    sess_config = tf.estimator.RunConfig(
        log_step_count_steps=500)

    ptb_word_lm = tf.estimator.Estimator(
        model_fn=model_function,
        config=sess_config,
        model_dir=FLAGS.save_path,
        params=config)

    for _ in range(config.max_max_epoch):
        ptb_word_lm.train(input_fn=train_input.input_data)
        ptb_word_lm.evaluate(input_fn=valid_input.input_data)
    ptb_word_lm.evaluate(input_fn=test_input.input_data)


if __name__ == "__main__":
    # tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="small",
        help="A type of model. Possible options are: small, medium, large.")

    parser.add_argument(
        "--data_path",
        type=str,
        help="Where the training/test data is stored.")

    parser.add_argument(
        "--save_path",
        type=str,
        help="Model output directory.")

    parser.add_argument(
        "--rnn_mode",
        type=str,
        default=None,
        help="The low level implementation of lstm cell: one of BASIC, "
             "or BLOCK, representing basic_lstm, or lstm_block_cell classes.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
