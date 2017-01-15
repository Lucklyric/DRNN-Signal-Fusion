import tensorflow as tf
import tensorflow.contrib.layers as tc
import numpy as np
import data_util

IS_TRAINING = 1
START_LEARNING_RATE = 0.0005


class Config(object):
    batch_size = 20
    time_steps = 20
    input_size = 22
    output_size = 2
    cell_size = 200
    dense_size = 200
    rnn_layer_size = 5
    keep_prob_dense = 0.2
    keep_prob_rnn = 0.6
    start_learning_rate = START_LEARNING_RATE


class TrainConfig(Config):
    batch_size = 1
    time_steps = 4300
    input_size = 11
    output_size = 2
    rnn_layer_size = 5
    cell_size = 200
    dense_size = 200
    keep_prob_dense = 0.2
    keep_prob_rnn = 0.6
    start_learning_rate = START_LEARNING_RATE


class TestConfig(TrainConfig):
    time_steps = 1
    keep_pro_dense = 1
    keep_pro_rnn = 1


class DRNNFusion(object):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.time_steps = config.time_steps
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.rnn_cell_size = config.cell_size
        self.keep_prob_dense = config.keep_prob_dense
        self.keep_prob_rnn = config.keep_prob_rnn
        self.start_learning_rate = config.start_learning_rate
        self.dense_size = config.dense_size
        self.rnn_layer_size = config.rnn_layer_size

        # Build the net
        with tf.variable_scope("inputs"):
            self.xs_sensor_1 = tf.placeholder(tf.float32, [self.batch_size, self.time_steps, self.input_size],
                                              name='xs_sensor_1')
            self.xs_sensor_2 = tf.placeholder(tf.float32, [self.batch_size, self.time_steps, self.input_size],
                                              name='xs_sensor_2')
            self.ys = tf.placeholder(tf.float32, [self.batch_size, self.time_steps, self.output_size], name='ys')

        with tf.variable_scope("configure"):
            self.keep_prob_rnn_tf = tf.placeholder(tf.float32, name="keep_prob_rnn")
            self.keep_prob_dense_tf = tf.placeholder(tf.float32, name="keep_prob_rnn_dense_tf")
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        with tf.name_scope('fc1'):
            with tf.variable_scope("fc1_variable"):
                # sensor 1
                fc1_inputs_sensor_1 = self.reshape_to_flat(self.xs_sensor_1, [-1, self.input_size])
                fc1_inputs_sensor_2 = self.reshape_to_flat(self.xs_sensor_2, [-1, self.input_size])

                fc1_sensor_1 = tc.layers.fully_connected(fc1_inputs_sensor_1, self.dense_size, activation_fn=tf.nn.relu,
                                                         weights_initializer=tc.initializers.xavier_initializer(),
                                                         )

                fc1_sensor_2 = tc.layers.fully_connected(fc1_inputs_sensor_2, self.dense_size, activation_fn=tf.nn.relu,
                                                         weights_initializer=tc.initializers.xavier_initializer(),
                                                         )

        with tf.name_scope('split_rnn'):
            with tf.variable_scope("split_rnn_variable_sensor_1") as scope:
                self.cell_init_state_sensor_1, self.cell_outputs_sensor_1, self.cell_final_state_sensor_1 = \
                    self.add_rnn_cell(self.reshape_to_three_d(fc1_sensor_1, [-1, self.time_steps, self.rnn_cell_size]),
                                      scope)

            with tf.variable_scope("split_rnn_variable_sensor_2") as scope:
                self.cell_init_state_sensor_2, self.cell_outputs_sensor_2, self.cell_final_state_sensor_2 = \
                    self.add_rnn_cell(self.reshape_to_three_d(fc1_sensor_2, [-1, self.time_steps, self.rnn_cell_size]),
                                      scope)
        with tf.name_scope('merge'):
            merged_outputs = tf.concat(1, [self.reshape_to_flat(self.cell_outputs_sensor_1, [-1, self.rnn_cell_size]),
                                           self.reshape_to_flat(self.cell_outputs_sensor_2, [-1, self.rnn_cell_size])])

        with tf.name_scope('output_fc2'):
            with tf.variable_scope("fc2_variable"):
                self.fc2 = tc.layers.fully_connected(merged_outputs, self.dense_size // 2, activation_fn=tf.nn.relu,
                                                     weights_initializer=tc.initializers.xavier_initializer(),
                                                     )
                self.outputs = tc.layers.fully_connected(merged_outputs, self.output_size, activation_fn=None,
                                                         weights_initializer=tc.initializers.xavier_initializer(),
                                                         )
        with tf.name_scope('cost'):
            losses = tf.nn.seq2seq.sequence_loss(
                [tf.reshape(self.outputs, [-1, self.output_size], name="reshape_pred")],
                [tf.reshape(self.ys, [-1, self.output_size], name="reshape_target")],
                [tf.ones([self.output_size, self.batch_size * self.time_steps], dtype=tf.float32)],
                average_across_timesteps=True,
                softmax_loss_function=tf.nn.softmax_cross_entropy_with_logits
            )
            with tf.name_scope('average_cost'):
                self.cost = tf.div(
                    tf.reduce_sum(losses, name='losses_sum'),
                    self.batch_size,
                    name='average_cost')
                tf.summary.scalar('cost', self.cost)

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        with tf.name_scope("correct_pred"):
            self.correct_pred = tf.arg_max(self.outputs, 1)

    def reshape_to_flat(self, inputs, shape):
        return tf.reshape(inputs, shape=shape, name="2_2D")

    def reshape_to_three_d(self, inputs, shape):
        return tf.reshape(inputs, shape=shape, name="2_3D")

    def add_rnn_cell(self, inputs, scope):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_cell_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob_rnn_tf)
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.rnn_layer_size)
        cell_init_state = stacked_lstm.zero_state(self.batch_size,
                                                  dtype=tf.float32)
        cell_outputs, cell_final_state = tf.nn.dynamic_rnn(
            stacked_lstm, inputs, initial_state=cell_init_state, time_major=False)
        return cell_init_state, cell_outputs, cell_final_state


def eval_accuracy(sess, model, data_set_instance, val_init_state):
    predict = []
    for j in range(len(data_set_instance._inputs)):
        if j == 0:
            feed_dict = {
                model.xs_sensor_1: (np.reshape(data_set_instance._inputs[j], [1, 1, 22]))[:, :, :11],
                model.xs_sensor_2: (np.reshape(data_set_instance._inputs[j], [1, 1, 22]))[:, :, 11:],
                model.keep_prob_rnn_tf: 1,
                model.keep_prob_dense_tf: 1,
                model.cell_init_state_sensor_1: val_init_state,
                model.cell_init_state_sensor_2: val_init_state,
                model.learning_rate: model.start_learning_rate
            }
        else:
            feed_dict = {
                model.xs_sensor_1: (np.reshape(data_set_instance._inputs[j], [1, 1, 22]))[:, :, :11],
                model.xs_sensor_2: (np.reshape(data_set_instance._inputs[j], [1, 1, 22]))[:, :, 11:],
                model.keep_prob_rnn_tf: 1,
                model.keep_prob_dense_tf: 1,
                model.cell_init_state_sensor_1: val_state_1,
                model.cell_init_state_sensor_2: val_state_2,
                model.learning_rate: model.start_learning_rate
            }
        pred, val_state_1, val_state_2 = sess.run(
            [model.pred, model.cell_final_state_sensor_1, model.cell_final_state_sensor_2], feed_dict=feed_dict)
        predict.append(pred[0])
    result, percent = data_set_instance.evaluate(predict)
    print("%f,%f" % (result, percent))
    return predict


def run_train(sess, model, train_data):
    epoch = 0
    epoch_cost = 0
    epoch_count = 0
    train_init_state = session.run(drnn_fusion_model_train.cell_init_state_sensor_1)
    state1 = state2 = None
    while epoch < 1000:
        batch_x, batch_y, is_reset = train_data.get_batch()
        if epoch == 0 and epoch_count == 0:
            feed_dict = {
                model.xs_sensor_1: batch_x[:, :, :11],
                model.xs_sensor_2: batch_x[:, :, 11:],
                model.ys: batch_y,
                model.keep_prob_rnn_tf: model.keep_prob_rnn,
                model.keep_prob_dense_tf: model.keep_prob_dense,
                model.learning_rate: model.start_learning_rate
            }
        else:
            if is_reset == 1:
                print("epoch_count %d, cost:%.4f" % (
                    epoch_count, round(epoch_cost / epoch_count, 4)))
                saver.save(sess, "tmp/model.ckpt")
                epoch += 1
                epoch_count = 0
                epoch_cost = 0
                feed_dict = {
                    model.xs_sensor_1: batch_x[:, :, :11],
                    model.xs_sensor_2: batch_x[:, :, 11:],
                    model.ys: batch_y,
                    model.keep_prob_rnn_tf: model.keep_prob_rnn,
                    model.keep_prob_dense_tf: model.keep_prob_dense,
                    model.cell_init_state_sensor_1: train_init_state,
                    model.cell_init_state_sensor_2: train_init_state,
                    model.learning_rate: model.start_learning_rate
                }
            else:
                feed_dict = {
                    model.xs_sensor_1: batch_x[:, :, :11],
                    model.xs_sensor_2: batch_x[:, :, 11:],
                    model.ys: batch_y,
                    model.keep_prob_rnn_tf: model.keep_prob_rnn,
                    model.keep_prob_dense_tf: model.keep_prob_dense,
                    model.cell_init_state_sensor_1: state1,
                    model.cell_init_state_sensor_2: state2,
                    model.learning_rate: model.start_learning_rate
                }
        _, cost, state1, state2, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state_sensor_1, model.cell_final_state_sensor_2,
             model.outputs],
            feed_dict=feed_dict
        )
        epoch_cost += cost
        epoch_count += 1
        # if epoch % 5 == 0 and is_reset == 1:
        #     eval_accuracy(sess, test_data_set)


if __name__ == "__main__":
    train_config = TrainConfig()
    drnn_fusion_model_train = DRNNFusion(train_config)
    session = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", session.graph)
    saver = tf.train.Saver()

    train_data_set, test_data_set = data_util.prepare_data(drnn_fusion_model_train.batch_size,
                                                           drnn_fusion_model_train.time_steps)
    init = tf.global_variables_initializer()
    session.run(init)

    ckpt = tf.train.get_checkpoint_state('tmp/')

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        print('Checkpoint found')
    else:
        print('No checkpoint found')
    if IS_TRAINING == 1:
        run_train(session, drnn_fusion_model_train, train_data_set)
    else:
        pass
