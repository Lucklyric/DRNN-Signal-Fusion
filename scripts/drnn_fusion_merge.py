import tensorflow as tf
import tensorflow.contrib.layers as tc
import numpy as np
import data_util

IS_TRAINING = 1
START_LEARNING_RATE = 0.0015


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
    is_training = True
    start_learning_rate = START_LEARNING_RATE


class TrainConfig(Config):
    batch_size = 1
    time_steps = 3000
    input_size = 11
    output_size = 2
    rnn_layer_size = 3
    cell_size = 256
    dense_size = 256
    keep_prob_dense = 0.5
    keep_prob_rnn = 0.5
    start_learning_rate = START_LEARNING_RATE


class TestConfig(TrainConfig):
    batch_size = 1
    time_steps = 1
    keep_pro_dense = 1
    keep_pro_rnn = 1
    is_training = False


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
        self.is_training = config.is_training

        # Build the net
        if self.is_training is True:
            name_prefix = "train"
        else:
            name_prefix = "test"

        print name_prefix
        with tf.variable_scope(name_prefix + "_inputs"):
            self.xs_sensor_1 = tf.placeholder(tf.float32, [self.batch_size, self.time_steps, self.input_size],
                                              name='xs_sensor_1')
            self.xs_sensor_2 = tf.placeholder(tf.float32, [self.batch_size, self.time_steps, self.input_size],
                                              name='xs_sensor_2')
            self.ys = tf.placeholder(tf.float32, [self.batch_size, self.time_steps, self.output_size], name='ys')

        with tf.variable_scope("configure"):
            self.keep_prob_rnn_tf = tf.placeholder(tf.float32, name="keep_prob_rnn")
            self.keep_prob_dense_tf = tf.placeholder(tf.float32, name="keep_prob_rnn_dense_tf")
            if self.is_training is True:
                self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            if self.is_training is True:
                tf.summary.scalar('lr', self.learning_rate)
        with tf.name_scope('input_fc'):
            with tf.variable_scope("fc_variable"):
                # sensor 1
                fc1_inputs_sensor_1 = self.reshape_to_flat(self.xs_sensor_1, [-1, self.input_size])
                fc1_inputs_sensor_2 = self.reshape_to_flat(self.xs_sensor_2, [-1, self.input_size])

                fc1_sensor_1 = self.add_fc_layer(fc1_inputs_sensor_1, self.dense_size, "fc_1",
                                                 self.keep_prob_dense_tf,
                                                 act_fn=tf.nn.relu, reuse=not self.is_training)
                # fc1_sensor_1 = self.add_fc_layer(fc1_s1_1, self.rnn_cell_size, True, self.keep_prob_dense_tf,
                #                                  name="fc1_s1_2")

                fc1_sensor_2 = self.add_fc_layer(fc1_inputs_sensor_2, self.dense_size, "fc_1",
                                                 self.keep_prob_dense_tf, act_fn=tf.nn.relu, reuse=True
                                                 )
                # fc1_sensor_2 = self.add_fc_layer(fc1_s2_1, self.rnn_cell_size, True, self.keep_prob_dense_tf,
                #                                  name="fc1_s2_2")

        with tf.name_scope('split_rnn'):
            # with tf.variable_scope("split_rnn_variable_sensor_1") as scope:
            self.cell_init_state_sensor_1, self.cell_outputs_sensor_1, self.cell_final_state_sensor_1 = \
                self.add_rnn_cell(self.reshape_to_three_d(fc1_sensor_1, [-1, self.time_steps, self.rnn_cell_size]),
                                  not self.is_training)

            # with tf.variable_scope("split_rnn_variable_sensor_2") as scope:
            self.cell_init_state_sensor_2, self.cell_outputs_sensor_2, self.cell_final_state_sensor_2 = \
                self.add_rnn_cell(self.reshape_to_three_d(fc1_sensor_2, [-1, self.time_steps, self.rnn_cell_size]),
                                  True)

        with tf.name_scope('merge'):
            merged_outputs = tf.concat(1, [self.reshape_to_flat(self.cell_outputs_sensor_1, [-1, self.rnn_cell_size]),
                                           self.reshape_to_flat(self.cell_outputs_sensor_2, [-1, self.rnn_cell_size])])

        with tf.name_scope('output_fc'):
            with tf.variable_scope("fc_variable"):
                self.fc2 = self.add_fc_layer(merged_outputs, self.dense_size, "fc_2_1", self.keep_prob_dense_tf,
                                             tf.nn.relu, reuse=not self.is_training)

                # fc_2_2 = self.add_fc_layer(self.fc2, self.dense_size, True, self.keep_prob_dense_tf, "fc_2_2")
                self.outputs = self.add_fc_layer(self.fc2, self.output_size, "fc_2_2", self.keep_prob_dense_tf,
                                                 None, reuse=not self.is_training)
        with tf.name_scope("correct_pred"):
            self.correct_pred = tf.arg_max(self.outputs, 1)
        if self.is_training is False:
            return
        with tf.name_scope('cost'):
            if self.is_training is False:
                tf.get_variable_scope().reuse_variables()
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

    def reshape_to_flat(self, inputs, shape):
        return tf.reshape(inputs, shape=shape, name="2_2D")

    def reshape_to_three_d(self, inputs, shape):
        return tf.reshape(inputs, shape=shape, name="2_3D")

    def add_fc_layer(self, inputs, outputs, name, keep_prob=None, act_fn=None, reuse=None):
        with tf.variable_scope(name) as scope:
            if reuse is True:
                tf.get_variable_scope().reuse_variables()
            fc = tc.layers.fully_connected(inputs, outputs, activation_fn=act_fn,
                                           weights_initializer=tc.initializers.xavier_initializer(),
                                           scope=scope, reuse=reuse)
            if self.is_training and keep_prob is not None:
                fc = tf.nn.dropout(fc, keep_prob)
        return fc

    def add_rnn_cell(self, inputs, reuse=None):
        if reuse is True:
            tf.get_variable_scope().reuse_variables()

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_cell_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob_rnn_tf)
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.rnn_layer_size)
        cell_init_state = stacked_lstm.zero_state(self.batch_size,
                                                  dtype=tf.float32)
        cell_outputs, cell_final_state = tf.nn.dynamic_rnn(
            stacked_lstm, inputs, initial_state=cell_init_state, time_major=False)
        return cell_init_state, cell_outputs, cell_final_state


def eval_accuracy(sess, model, data_set_instance, val_init_state, mode=1):
    predict = []
    batch_x, _, _ = data_set_instance.get_batch()
    if mode == 2:
        feed_dict = {
            model.xs_sensor_1: batch_x[:, :, :11],
            model.xs_sensor_2: batch_x[:, :, 11:],
            model.keep_prob_rnn_tf: 1,
            model.keep_prob_dense_tf: 1,
            model.cell_init_state_sensor_1: val_init_state,
            model.cell_init_state_sensor_2: val_init_state,
        }
        pred, val_state_1, val_state_2 = sess.run(
            [model.correct_pred, model.cell_final_state_sensor_1,
             model.cell_final_state_sensor_2], feed_dict=feed_dict)
        result, percent = data_set_instance.evaluate(pred.tolist())
        print("GE: %f,Accuracy:%f" % (result, percent))
        return predict
    for j in range(len(data_set_instance._inputs)):
        if j == 0:
            feed_dict = {
                model.xs_sensor_1: (np.reshape(data_set_instance._inputs[j], [1, 1, 22]))[:, :, :11],
                model.xs_sensor_2: (np.reshape(data_set_instance._inputs[j], [1, 1, 22]))[:, :, 11:],
                model.keep_prob_rnn_tf: 1,
                model.keep_prob_dense_tf: 1,
                model.cell_init_state_sensor_1: val_init_state,
                model.cell_init_state_sensor_2: val_init_state,
            }
        else:
            feed_dict = {
                model.xs_sensor_1: (np.reshape(data_set_instance._inputs[j], [1, 1, 22]))[:, :, :11],
                model.xs_sensor_2: (np.reshape(data_set_instance._inputs[j], [1, 1, 22]))[:, :, 11:],
                model.keep_prob_rnn_tf: 1,
                model.keep_prob_dense_tf: 1,
                model.cell_init_state_sensor_1: val_state_1,
                model.cell_init_state_sensor_2: val_state_2,
            }
        pred, val_state_1, val_state_2 = sess.run(
            [model.correct_pred, model.cell_final_state_sensor_1,
             model.cell_final_state_sensor_2], feed_dict=feed_dict)
        predict.append(pred[0])
    result, percent = data_set_instance.evaluate(predict)
    print("GE: %f,Accuracy:%f" % (result, percent))
    return predict


def run_train(sess, model, train_data):
    epoch = 0
    epoch_cost = 0
    epoch_count = 0
    train_init_state = session.run(model.cell_init_state_sensor_1)
    state1 = state2 = None
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    previous_losses = []

    lr = model.start_learning_rate
    while epoch < 1000:
        batch_x, batch_y, is_reset = train_data.get_batch()
        if epoch == 0 and epoch_count == 0:
            feed_dict = {
                model.xs_sensor_1: batch_x[:, :, :11],
                model.xs_sensor_2: batch_x[:, :, 11:],
                model.ys: batch_y,
                model.keep_prob_rnn_tf: model.keep_prob_rnn,
                model.keep_prob_dense_tf: model.keep_prob_dense,
                model.learning_rate: lr
            }
        else:
            if is_reset == 1:
                print("epoch: %d, cost:%.4f, learning_rate:%f" % (
                    epoch, round(epoch_cost / epoch_count, 4), lr))
                saver.save(sess, "tmp/model.ckpt")
                writer.add_summary(cost_summary, epoch)
                if len(previous_losses) > 4.0 and epoch_cost > max(previous_losses[-5:]) and lr > 0.00002:
                    print ("decay learning rate!!!")
                    lr /= 2.0
                previous_losses.append(epoch_cost)
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
                    model.learning_rate: lr
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
                    model.learning_rate: lr
                }
        _, cost, state1, state2, pred, cost_summary, lr = sess.run(
            [model.train_op, model.cost, model.cell_final_state_sensor_1, model.cell_final_state_sensor_2,
             model.outputs, merged, model.learning_rate],
            feed_dict=feed_dict
        )
        epoch_cost += cost
        epoch_count += 1
        if epoch % 1 == 0 and is_reset == 1:
            eval_accuracy(sess, drnn_fusion_model_evl, test_data_set, train_init_state, 1)


def run_eval(sess, model, test_data, train_init_state=None):
    print(test_data.cal_worst_best())
    if train_init_state is None:
        train_init_state = session.run(model.cell_init_state_sensor_1)
    predict = eval_accuracy(sess, model, test_data, train_init_state)
    print(predict)


if __name__ == "__main__":
    with tf.variable_scope('model') as scope:
        if IS_TRAINING == 1:
            train_config = TrainConfig()
            drnn_fusion_model = DRNNFusion(train_config)
            drnn_fusion_model_evl = DRNNFusion(TestConfig())
            _, test_data_set = data_util.prepare_data(drnn_fusion_model_evl.batch_size,
                                                      drnn_fusion_model_evl.time_steps)
            train_data_set, _ = data_util.prepare_data(drnn_fusion_model.batch_size,
                                                       drnn_fusion_model.time_steps)
        else:
            test_config = TestConfig()
            drnn_fusion_model = DRNNFusion(test_config)
            train_data_set, test_data_set = data_util.prepare_data(drnn_fusion_model.batch_size,
                                                                   drnn_fusion_model.time_steps)
        session = tf.Session()

        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        session.run(init)
        print (drnn_fusion_model.time_steps)
        ckpt = tf.train.get_checkpoint_state('tmp/')

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            print('Checkpoint found')
        else:
            print('No checkpoint found')
        if IS_TRAINING == 1:
            run_train(session, drnn_fusion_model, train_data_set)
        else:
            run_eval(session, drnn_fusion_model, test_data_set)
