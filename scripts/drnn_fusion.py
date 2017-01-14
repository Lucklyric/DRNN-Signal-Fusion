from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import tensorflow.contrib.layers as tc
import data_util

BATCH_START = 0
TIME_STEPS = 1
BATCH_SIZE = 1
INPUT_SIZE = 22
OUTPUT_SIZE = 2
CELL_SIZE = 128
LR = 0.000025
IS_TRAINING = 1


class LSTMRNN(object):
    def __init__(self, n_step, input_size, output_size, cell_size, batch_size):
        self._n_steps = n_step
        self._input_size = input_size
        self._output_size = output_size
        self._cell_size = cell_size
        self._batch_size = batch_size
        with tf.name_scope("inputs"):
            self.xs = tf.placeholder(tf.float32, [None, self._n_steps, input_size], name="xs")
            self.ys = tf.placeholder(tf.float32, [None, self._n_steps, output_size], name="ys")
            self.keep_prob = tf.placeholder(tf.float32)
        with tf.variable_scope("in_hidden"):
            self.add_input_layer()
        with tf.variable_scope("LSTM_cell"):
            self.add_cell()
        with tf.variable_scope("out_hidden"):
            self.add_output_layer()
        with tf.name_scope("cost"):
            self.comput_cost()
        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)
        with tf.name_scope("correct_pred"):
            self.pred = tf.arg_max(self.pred, 1)
            # self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            # with tf.name_scope("reset_state"):
            #     self.reset_state = tf.assign(self.cell_init_state, self.cell_zero_state)
            # with tf.name_scope("process_val"):
            #     self.switch_val_state = tf.assign(self.cell_init_state, self.cell_zero_state_val)

    def add_input_layer(self):
        l_in_x = tf.reshape(self.xs, [-1, self._input_size], name="2_2D")
        Ws_in = self._weight_variable([self._input_size, self._cell_size])
        bs_in = self._bias_variable([self._cell_size, ])

        with tf.name_scope("Wx_plus_b"):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
            self.l_in_y_shaped = tf.reshape(l_in_y, [-1, self._n_steps, self._cell_size], name="2_3D")
            # first_hidden_layer = (tc.relu(l_in_y, 200))
            # second_hidden_layer = (tc.relu(first_hidden_layer, 200))
            # third_hidden_layer = tf.nn.dropout(tc.relu(second_hidden_layer, self._cell_size), self.keep_prob)
            # self.third_hidden_layer_y = tf.reshape(third_hidden_layer, [-1, self._n_steps, self._cell_size], name="2_3D")

    def add_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self._cell_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.6)
        self.stacked_lstm = rnn_cell.MultiRNNCell([lstm_cell] * 5,
                                                  state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = self.stacked_lstm.zero_state(self._batch_size,
                                                                dtype=tf.float32)
            self.cell_init_state_val = self.stacked_lstm.zero_state(1,
                                                                    dtype=tf.float32)

        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            self.stacked_lstm, self.l_in_y_shaped, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        l_out_x = tf.reshape(self.cell_outputs, [-1, self._cell_size], name='2_2D')
        fourth_hidden_layer = tf.nn.dropout(tc.relu(l_out_x, self._cell_size), self.keep_prob)
        # fifth_hidden_layer = tf.nn.dropout(tc.relu(fourth_hidden_layer, self._cell_size), self.keep_prob)
        Ws_out = self._weight_variable([self._cell_size, self._output_size])
        bs_out = self._bias_variable([self._output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(fourth_hidden_layer, Ws_out) + bs_out

    def comput_cost(self):
        losses = tf.nn.seq2seq.sequence_loss(
            [tf.reshape(self.pred, [-1, 2], name="reshape_pred")],
            [tf.reshape(self.ys, [-1, 2], name="reshape_target")],
            [tf.ones([2, self._batch_size * self._n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=tf.nn.softmax_cross_entropy_with_logits
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self._batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


def eval_accuracy(session, data_set_instance):
    predict = []
    for j in range(len(data_set_instance._inputs)):
        if j == 0:
            feed_dict = {
                model.xs: np.reshape(data_set_instance._inputs[j], [1, 1, 22]),
                model.cell_init_state: val_init_state,
                model.keep_prob: 0.8
            }
        else:
            feed_dict = {
                model.xs: np.reshape(data_set_instance._inputs[j], [1, 1, 22]),
                model.cell_init_state: valid_state,
                model.keep_prob: 0.8
            }
        pred, valid_state = sess.run([model.pred, model.cell_final_state], feed_dict=feed_dict)
        predict.append(pred[0])
    result, percent = data_set_instance.evaluate(predict)
    print("%f,%f" % (result, percent))
    return predict


def run_eval():
    print(test_data.cal_worst_best())
    predict = eval_accuracy(sess, test_data)
    print(predict)


def run_training():
    epoch = 0
    epoch_cost = 0
    epoch_count = 0
    while epoch < 1000:
        batch_x, batch_y, is_reset = train_data.get_batch()
        if epoch == 0 and epoch_count == 0:
            feed_dict = {
                model.xs: batch_x,
                model.ys: batch_y,
                model.keep_prob: 0.2
            }
        else:
            if is_reset == 1:
                print("epoch_count %d, cost:%.4f, %d" % (
                    epoch_count, round(epoch_cost / epoch_count, 4), train_data._start_cursor))
                saver.save(sess, "tmp/model.ckpt")
                epoch += 1
                epoch_count = 0
                epoch_cost = 0
                feed_dict = {
                    model.xs: batch_x,
                    model.ys: batch_y,
                    model.cell_init_state: train_init_state,
                    model.keep_prob: 0.2
                }
            else:
                feed_dict = {
                    model.xs: batch_x,
                    model.ys: batch_y,
                    model.cell_init_state: state,
                    model.keep_prob: 0.2
                }
        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict
        )
        epoch_cost += cost
        epoch_count += 1
        if epoch % 10 == 0 and is_reset == 1:
            eval_accuracy(sess, test_data)
            # count = 0
            # true = 0
            # for j in range(len(test_data._inputs)):
            #     if j == 0:
            #         feed_dict = {
            #             model.xs: np.reshape(test_data._inputs[j], [1, 1, 22]),
            #             model.cell_init_state: val_init_state
            #         }
            #     else:
            #         feed_dict = {
            #             model.xs: np.reshape(test_data._inputs[j], [1, 1, 22]),
            #             model.cell_init_state: valid_state
            #         }
            #     pred, valid_state = sess.run([model.pred, model.cell_final_state], feed_dict=feed_dict)
            #     if test_data._outputs[j][pred[0]] == 1:
            #         true += 1
            #     count += 1
            # print("%d,%d,%f" % (true, count, true / count))


if __name__ == "__main__":
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    saver = tf.train.Saver()

    train_data, test_data = data_util.prepare_data(BATCH_SIZE, TIME_STEPS)
    init = tf.global_variables_initializer()
    sess.run(init)
    train_init_state = sess.run(model.cell_init_state)
    val_init_state = sess.run(model.cell_init_state_val)
    ckpt = tf.train.get_checkpoint_state('tmp/')

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Checkpoint found')
    else:
        print('No checkpoint found')
    if IS_TRAINING == 0:
        run_eval()
    else:
        run_training()
