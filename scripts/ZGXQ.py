import tensorflow as tf
import tensorflow.contrib.layers as tc
import numpy as np

with tf.name_scope("inputs"):
    board_input = tf.placeholder(tf.float32, [None, 90], "Board_Input")
    board_label = tf.placeholder(tf.float32, [None, 10000], "Board_Label")

with tf.name_scope("inputs_reshape"):
    board_input_reshape = tf.reshape(board_input, [-1, 10, 9, 1], name="board_input_reshape")

with tf.name_scope("conv1"):
    conv1 = tc.conv2d(board_input_reshape, 30, kernel_size=[1, 2, 2, 1])
    conv1_pool = tc.max_pool2d(conv1, kernel_size=[1, 2, 2, 1], stride=[1, 2, 2, 1])

with tf.name_scope("conv2"):
    conv2 = tc.conv2d(conv1_pool, 64, kernel_size=[1, 2, 2, 1])
    conv2_pool = tc.max_pool2d(conv2, kernel_size=[1, 2, 2, 1], stride=[1, 2, 2, 1])

with tf.name_scope("fc1"):
    fc1 = tf.nn.dropout(tc.fully_connected(conv2_pool, 5000), 0.5)

with tf.name_scope("output"):
    board_output = tc.fully_connected(fc1, 10000, activation_fn=None)

with tf.name_scope("loss"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(board_label, board_output))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(board_output, 1), tf.argmax(board_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("logs-ZGXQ",sess.graph)
# train_step.run(feed_dict={x=batch[0]})
