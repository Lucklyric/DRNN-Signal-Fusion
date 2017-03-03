import data_util
import numpy as np

# import tensorflow as tf

train_data, test_data = data_util.prepare_data(1, 1)
# print (np.shape(train_data.get_batch()))
# data_x,data_y,_ = test_data.get_batch()
# print(np.reshape(train_data._inputs[0], [1, 1, 22]))
# print(test_data.best_error_mean, test_data.worst_error_mean, test_data._s1_error_mean, test_data._s2_error_mean)
# print(test_data.get_batch())
# print(test_data.get_batch())
x_val = test_data._outputs;
length = np.shape(x_val)[0]
sum_mean_error = 0
for epoch in range(100):
    randmon_y=np.random.randint(2, size=length)
    error,_=test_data.evaluate(randmon_y)
    print (error)
    sum_mean_error = sum_mean_error + error

print("mean:%f" % (sum_mean_error/100))
print("accuracy:%f" % (1-(sum_mean_error/100-test_data.best_error_mean)/
                       (test_data.worst_error_mean-test_data.best_error_mean)))
print("B:%f,W:%f" %(test_data.best_error_mean,test_data.worst_error_mean))
# print length
# print ("Best:%f,Worst:%f"%(test_data.best_error_mean,test_data.worst_error_mean))
# print (np.shape(train_data._inputs))
# is_reset = 0
# while is_reset == 0:
#     print (train_data._start_cursor)
#     _, _, is_reset = train_data.get_batch()
# data_x,data_y,_ = test_data.get_batch()
# print (data_x)
# dx = tf.placeholder(shape=[None, 22], dtype=tf.float32)
# x_split = tf.split(5, 2, dx)
#
# with tf.session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print sess.run(x_split, feed_dict={dx: data_x})
