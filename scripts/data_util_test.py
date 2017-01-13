import data_util
import numpy as np
# import tensorflow as tf

train_data, test_data = data_util.prepare_data(2, 2)
# print (np.shape(train_data.get_batch()))
data_x,data_y,_ = test_data.get_batch()
# print(np.reshape(train_data._inputs[0], [1, 1, 22]))
print(train_data._outputs)
# data_x,data_y,_ = test_data.get_batch()
# print (data_x)
# dx = tf.placeholder(shape=[None, 22], dtype=tf.float32)
# x_split = tf.split(5, 2, dx)
#
# with tf.session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print sess.run(x_split, feed_dict={dx: data_x})
